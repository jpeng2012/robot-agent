"""Robosuite backend worker. Runs under the robosuite conda interpreter (Python 3.10), NOT robot-agent's.

Speaks JSON lines on stdout: one request per stdin line, one reply per stdout line. It must not
import robot_agent; replies are plain dicts shaped like robot_agent.world.{WorldState,SkillOutcome}.

Scene and controller: robot-policy-learning's ConstrainedPickPlace (Lift + barrier + visual target
marker), Panda with robosuite's default OSC controller at 10 Hz. Skills run segments of
constrained_pick_place_expert_action: pick = APPROACH..LIFT, place = TRANSPORT..DONE.

Agent frame: x, y are world x, y; z = world z - z_ref, where z_ref is the cube's resting centre
height (table top + cube half-height). An object resting on the table therefore has z ~= 0,
matching the mock's convention, and the target region is reported at z = 0.
"""

import json
import os
import sys
import traceback

import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler

from envs.constrained_pick_place import ConstrainedPickPlace
from experts.constrained_pick_place_expert import constrained_pick_place_expert_action

GRASP_TOL = 0.02  # same as robot_agent.env.GRASP_TOL
PLACE_TOL = 0.02  # same as robot_agent.skills.PLACE_TOL
LIFTED_Z = 0.08  # agent-frame cube height that counts as lifted (expert lifts to ~0.10)
REST_Z_TOL = 0.025  # |z| that counts as resting on the table (existing success check uses 0.025)
PICK_BUDGET = 150  # control steps (10 Hz)
PLACE_BUDGET = 450  # barrier detour alone takes up to ~270 steps
GRASP_PATIENCE = 20  # steps in GRASP without a stable grasp before giving up
RETREAT_Z = 0.12  # agent-frame eef height to retreat to after place / failed pick
# Reachable placement region (world xy). Matches the expert's validated cube / target ranges with margin.
REACH = (-0.15, 0.15, -0.12, 0.36)
BARRIER_MARGIN = 0.05


def _l(v):
    return [float(x) for x in v]


class Backend:
    def __init__(self, render=False):
        sampler = UniformRandomSampler(  # same sampler as the demo collector; reset() overrides the pose
            name="ObjectSampler", x_range=[-0.08, 0.08], y_range=[-0.05, 0.02], rotation=None,
            ensure_object_boundary_in_range=True, ensure_valid_placement=True,
            reference_pos=[0, 0, 0.8], z_offset=0.01,
        )
        self.env = ConstrainedPickPlace(
            robots="Panda", placement_initializer=sampler, has_renderer=render,
            has_offscreen_renderer=False, use_camera_obs=False, control_freq=10,
            horizon=100_000, ignore_done=True,  # episode length is bounded by the agent loop, not robosuite
        )
        assert self.env.action_dim == 7, self.env.action_dim
        self.render = render

    # ---------------------------------------------------------------- state

    def info(self):
        e = self.env
        return {
            "z_ref": self.z_ref, "table_top_z": float(e.table_offset[2]),
            "cube_half_size": _l(e.cube.size), "target_half_size": _l(e.target_half_size),
            "barrier_pos": _l(e.barrier_pos), "barrier_half_size": _l(e.barrier_half_size),
            "control_freq": e.control_freq, "controller": type(e.robots[0].composite_controller).__name__,
        }

    def _grasped(self):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper,
                                          object_geoms=self.env.cube.contact_geoms))

    def _agent(self, p):
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2] - self.z_ref)}

    def _in_target(self, cube):
        hx, hy = self.env.target_half_size
        tx, ty = self.env.target_pos[:2]
        return bool(abs(cube[0] - tx) < hx and abs(cube[1] - ty) < hy)

    def reset(self, layout, seed=None):
        cubes = [k for k, (kind, _, _) in layout.items() if kind == "cube"]
        regions = [k for k, (kind, _, _) in layout.items() if kind == "target_region"]
        if len(cubes) != 1 or len(regions) != 1 or len(layout) != 2:
            raise ValueError(f"robosuite backend supports exactly one cube and one target_region, got {layout}")
        if seed is not None:
            np.random.seed(seed)
        self.cube_id, self.region_id = cubes[0], regions[0]
        e = self.env
        e.reset()
        self.z_ref = float(e.table_offset[2] + e.cube.size[2])
        _, cx, cy = layout[self.cube_id]
        _, tx, ty = layout[self.region_id]
        e.sim.data.set_joint_qpos(e.cube.joints[0], np.array([cx, cy, self.z_ref, 1.0, 0.0, 0.0, 0.0]))
        e.set_target([tx, ty, self.z_ref])  # calls sim.forward()
        self._settle(5)
        self.t0 = float(e.sim.data.time)
        self.step = 0
        self.ctx = None
        return self.observe()

    def observe(self):
        obs = self.obs
        cube = obs["cube_pos"]
        t = round(float(self.env.sim.data.time) - self.t0, 4)
        objects = {
            self.cube_id: {"id": self.cube_id, "kind": "cube", "pose": self._agent(cube),
                           "visible": True, "confidence": 1.0, "observed_at": t},
            self.region_id: {"id": self.region_id, "kind": "target_region",
                             "pose": {"x": float(self.env.target_pos[0]), "y": float(self.env.target_pos[1]), "z": 0.0},
                             "visible": True, "confidence": 1.0, "observed_at": t},
        }
        holding = self.cube_id if self._grasped() else None
        return {"t": t, "step": self.step, "objects": objects,
                "gripper": {"pose": self._agent(obs["robot0_eef_pos"]), "holding": holding}}

    # ---------------------------------------------------------------- control

    def _act(self, action):
        self.obs = self.env.step(action)[0]
        if self.render:
            self.env.render()

    def _settle(self, n, gripper=-1.0):
        self.obs = self.env._get_observations(force_update=True)
        for _ in range(n):
            self._act(np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32))

    def _retreat(self, max_steps=25):
        """Open the gripper and raise the eef straight up so it neither holds nor occludes the cube."""
        n = 0
        while n < max_steps and self.obs["robot0_eef_pos"][2] - self.z_ref < RETREAT_Z:
            a = np.zeros(7, dtype=np.float32)
            a[2], a[6] = 0.5, -1.0
            self._act(a)
            n += 1
        return n

    def _report(self, termination, fsm_state, steps, ok, code=None, detail=""):
        cube = self.obs["cube_pos"]
        info = {
            "control_steps": steps, "termination": termination, "fsm_state": fsm_state,
            "grasped": self._grasped(), "cube_pose": self._agent(cube), "cube_pose_world": _l(cube),
            "eef_pose_world": _l(self.obs["robot0_eef_pos"]), "in_target_region": self._in_target(cube),
        }
        out = {"ok": ok, "info": info}
        if not ok:
            out["failure"] = {"code": code, "detail": detail}
        return out

    def execute(self, name, args):
        self.step += 1
        if name == "pick":
            return self._pick(args["object_id"], args["x"], args["y"])
        if name == "place":
            return self._place(args["x"], args["y"])
        return {"ok": False, "failure": {"code": "UNKNOWN_TOOL", "detail": f"robosuite env has no skill {name!r}"}}

    def _reach_failure(self, x, y):
        x0, x1, y0, y1 = REACH
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return {"code": "OUT_OF_WORKSPACE",
                    "detail": f"({x:.3f}, {y:.3f}) outside robosuite reach x∈[{x0}, {x1}], y∈[{y0}, {y1}]"}
        bx, by, _ = self.env.barrier_pos
        hx, hy, _ = self.env.barrier_half_size
        if abs(x - bx) < hx + BARRIER_MARGIN and abs(y - by) < hy + BARRIER_MARGIN:
            return {"code": "OUT_OF_WORKSPACE", "detail": f"({x:.3f}, {y:.3f}) is on the barrier"}
        return None

    def _pick(self, oid, x, y):
        if oid != self.cube_id:
            return {"ok": False, "failure": {"code": "UNKNOWN_OBJECT", "detail": f"{oid!r} is not graspable"}}
        if self._grasped():
            return {"ok": False, "failure": {"code": "ALREADY_HOLDING", "detail": f"already holding {oid}"}}
        if (f := self._reach_failure(x, y)) is not None:
            return {"ok": False, "failure": f}
        target = np.array([x, y, self.z_ref])  # pick only needs a height reference for LIFT
        commanded = np.array([x, y, self.z_ref])
        ctx = {"grasped": False, "grasp_offset": None, "grasp_count": 0, "route": None, "transport_stage": 0}
        state, steps, grasp_steps, established = "APPROACH", 0, 0, False
        termination = "step_budget"
        while steps < PICK_BUDGET:
            grasped = self._grasped()
            if established and not grasped:
                termination = "grasp_lost"
                break
            ctx["grasped"] = grasped
            obs = dict(self.obs)
            if state in ("APPROACH", "DESCEND"):
                # Servo to the commanded point, not the privileged true cube pose.
                obs["cube_pos"] = commanded
            prev = state
            action, state = constrained_pick_place_expert_action(obs, state, target, ctx)
            if prev == "GRASP" and state == "LIFT":
                established = True  # expert saw a stable grasp for 5 consecutive steps
            self._act(action)
            steps += 1
            if state == "GRASP":
                grasp_steps += 1
                if grasp_steps > GRASP_PATIENCE:
                    termination = "no_grasp_contact"
                    break
            if state == "TRANSPORT":
                termination = "fsm_lifted"
                break
        self.ctx = ctx
        cube = self.obs["cube_pos"]
        grasped = self._grasped()
        lifted = cube[2] - self.z_ref >= LIFTED_Z
        if termination == "fsm_lifted" and grasped and lifted:
            return self._report(termination, state, steps, True)
        # Failure: leave the arm in a safe pose before reporting.
        steps += self._retreat()
        err = float(np.hypot(cube[0] - x, cube[1] - y))
        if established and not grasped:
            return self._report(termination, state, steps, False, "GRASP_SLIP",
                                f"{oid} slipped out of the gripper during {state} (established grasp lost)")
        if not established and err > GRASP_TOL:
            return self._report(termination, state, steps, False, "OBJECT_NOT_AT_POSE",
                                f"closed gripper at ({x:.3f}, {y:.3f}) but {oid} is {err:.3f} m away")
        return self._report(termination, state, steps, False, "POSTCONDITION_FAILED",
                            f"pick ended ({termination}, fsm={state}) with grasped={grasped}, "
                            f"cube z={cube[2] - self.z_ref:.3f} (need >= {LIFTED_Z})")

    def _place(self, x, y):
        if not self._grasped() or self.ctx is None or self.ctx.get("grasp_offset") is None:
            return {"ok": False, "failure": {"code": "NOT_HOLDING", "detail": "gripper is empty"}}
        if (f := self._reach_failure(x, y)) is not None:
            return {"ok": False, "failure": f}
        ctx = self.ctx
        bx, by, _ = self.env.barrier_pos
        cube = self.obs["cube_pos"]
        crosses = (cube[1] - by) * (y - by) < 0
        if crosses and cube[1] > by:
            return {"ok": False, "failure": {"code": "OUT_OF_WORKSPACE",
                                             "detail": "transport from beyond the barrier back toward the robot is unsupported"}}
        # The expert routes around the barrier (stages 0-1) only when the path crosses it; otherwise
        # go straight over to the target (stage 2).
        ctx["route"] = "LEFT" if (cube[0] + x) / 2 <= bx else "RIGHT"
        ctx["transport_stage"] = 0 if crosses else 2
        target = np.array([x, y, self.z_ref])
        state, steps, termination = "TRANSPORT", 0, "step_budget"
        while steps < PLACE_BUDGET:
            grasped = self._grasped()
            if state in ("TRANSPORT", "LOWER") and not grasped:
                termination = "grasp_lost"
                break
            ctx["grasped"] = grasped
            action, state = constrained_pick_place_expert_action(self.obs, state, target, ctx)
            self._act(action)
            steps += 1
            if state == "DONE":
                termination = "fsm_released"
                break
        steps += self._retreat()
        self._settle(5)
        steps += 5
        self.ctx = None
        cube = self.obs["cube_pos"]
        grasped = self._grasped()
        err = float(np.hypot(cube[0] - x, cube[1] - y))
        z = float(cube[2] - self.z_ref)
        if termination == "grasp_lost":
            return self._report(termination, state, steps, False, "GRASP_SLIP",
                                f"cube dropped during {state}; now at ({cube[0]:.3f}, {cube[1]:.3f}, z={z:.3f})")
        if not grasped and err <= PLACE_TOL and abs(z) <= REST_Z_TOL:
            return self._report(termination, state, steps, True)
        return self._report(termination, state, steps, False, "POSTCONDITION_FAILED",
                            f"place ended ({termination}, fsm={state}) with grasped={grasped}, cube "
                            f"{err:.3f} m from ({x:.3f}, {y:.3f}) at z={z:.3f}")


def main():
    # Keep the protocol channel clean: anything robosuite/MuJoCo prints goes to stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    backend = None
    for line in sys.stdin:
        req = json.loads(line)
        op = req["op"]
        try:
            if op == "init":
                backend = Backend(render=req.get("render", False))
                result = None
            elif op == "reset":
                result = backend.reset(req["layout"], req.get("seed"))
            elif op == "observe":
                result = backend.observe()
            elif op == "execute":
                result = backend.execute(req["name"], req["args"])
            elif op == "info":
                result = backend.info()
            elif op == "close":
                proto.write(json.dumps({"ok": True, "result": None}) + "\n")
                break
            else:
                raise ValueError(f"unknown op {op!r}")
            proto.write(json.dumps({"ok": True, "result": result}) + "\n")
        except Exception:
            proto.write(json.dumps({"ok": False, "error": traceback.format_exc()}) + "\n")
    if backend is not None:
        backend.env.close()


if __name__ == "__main__":
    main()
