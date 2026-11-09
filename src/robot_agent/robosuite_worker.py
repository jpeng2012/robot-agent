"""Robosuite backend worker. Runs under the robosuite conda interpreter (Python 3.10), NOT robot-agent's.

Speaks JSON lines on stdout: one request per stdin line, one reply per stdout line. It must not
import robot_agent; replies are plain dicts shaped like robot_agent.world.{WorldState,SkillOutcome}.

Scene and controller: robot-policy-learning's ConstrainedPickPlace (Lift + barrier + visual target
marker), Panda with robosuite's default OSC controller at 10 Hz. Skills run segments of
constrained_pick_place_expert_action: pick = APPROACH..LIFT, place = TRANSPORT..DONE.

Agent frame: x, y are world x, y; z = world z - z_ref, where z_ref is the cube's resting centre
height (table top + cube half-height). An object resting on the table therefore has z ~= 0,
matching the mock's convention, and the target region is reported at z = 0.

scene="two_cube" (F015): the same scene class with the barrier moved off the table (outside REACH, so no
transport ever crosses it), a second (blue) cube and a second region marker. Cubes are identified by
their simulator bodies (a red-named id gets the red Lift cube), each with its own resting-height reference.
Pick first travels at retreat height to above the commanded point and descends vertically; place refuses
points closer than PLACE_CLEARANCE to another resting cube (open-finger clearance). A test-only
perturbation schedule (schedule_perturb) moves a resting cube at the start of a chosen execute call, after
the client's validation and grounding, and is validated before anything is changed (transactional).
Implemented by TwoCubeBackend; the single-cube Backend is unchanged.
"""

import hashlib
import json
import os
import platform
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.models.objects import BoxObject
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
# two_cube scene (F015)
OFF_TABLE_BARRIER = (0.0, 0.9, 0.92)  # beyond the table edge (y <= 0.4) and far outside REACH
CUBE_B_HALF = 0.021  # fixed size (BoxObject size=...): no draws from env.rng
REGION_HALF = 0.045  # same as world.REGION_HALF; both markers use the scene's target_half_size
CUBE_OVERLAP = 0.04  # 2 * world.CUBE_HALF
PLACE_CLEARANCE = 0.09  # min centre distance to another resting cube for an open-finger place
TRANSIT_BUDGET = 80
TRANSIT_XY_TOL = 0.01


def _l(v):
    return [float(x) for x in v]



def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _versions():
    import mujoco
    import robosuite
    import envs.constrained_pick_place as env_mod
    import experts.constrained_pick_place_expert as expert_mod
    files = {"worker": os.path.abspath(__file__), "env": env_mod.__file__, "expert": expert_mod.__file__}
    return {
        "python": platform.python_version(), "robosuite": robosuite.__version__,
        "mujoco": mujoco.__version__, "numpy": np.__version__,
        "files": {k: {"path": v, "sha256": _sha256(v)} for k, v in files.items()},
    }

class TwoCubeScene(ConstrainedPickPlace):
    """ConstrainedPickPlace + a second cube (cube_b, blue) + a second visual region marker."""

    def _load_model(self):
        super()._load_model()
        self.cube_b = BoxObject(name="cube_b", size=[CUBE_B_HALF] * 3, rgba=[0.1, 0.3, 1.0, 1.0], rng=self.rng)
        body = self.cube_b.get_obj()
        body.set("pos", "0.3 0.35 0.9")  # parked clear of the robot until reset() places it
        self.model.merge_objects([self.cube_b])
        marker = ET.Element("body", attrib={"name": "region2_marker", "pos": "0.0 -0.3 0.802"})
        ET.SubElement(marker, "geom", attrib={
            "name": "region2_marker_geom", "type": "box",
            "size": self._vec_to_str([self.target_half_size[0], self.target_half_size[1], 0.002]),
            "rgba": "0.9 0.6 0.1 0.7", "group": "1", "contype": "0", "conaffinity": "0"})
        self.model.worldbody.append(marker)

    def _setup_references(self):
        super()._setup_references()
        self.cube_b_body_id = self.sim.model.body_name2id(self.cube_b.root_body)


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
        self.init = None  # realized initial conditions of the last reset
        self.n_resets = 0

    # ---------------------------------------------------------------- state

    def info(self):
        e = self.env
        return {
            "z_ref": self.z_ref, "table_top_z": float(e.table_offset[2]),
            "cube_half_size": _l(e.cube.size), "target_half_size": _l(e.target_half_size),
            "barrier_pos": _l(e.barrier_pos), "barrier_half_size": _l(e.barrier_half_size),
            "control_freq": e.control_freq, "controller": type(e.robots[0].composite_controller).__name__,
            "init": self.init, "versions": _versions(),
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
        e = self.env
        # robosuite 1.5.2 does not use the global numpy RNG: the cube size (drawn in _load_model on every
        # hard reset) and robot init-joint noise (_reset_internal) come from env.rng, and the placement
        # sampler has its own generator. Give both fresh generators on every reset: seeded, independent
        # streams when a seed is given, unseeded otherwise (so seed=None never replays an earlier seed).
        if seed is not None:
            np.random.seed(seed)
            env_ss, sampler_ss = np.random.SeedSequence(seed).spawn(2)
            e.rng, e.placement_initializer.rng = np.random.default_rng(env_ss), np.random.default_rng(sampler_ss)
        else:
            e.rng, e.placement_initializer.rng = np.random.default_rng(), np.random.default_rng()
        e.seed = seed
        self.n_resets += 1
        rng_meta = {  # which generators this reset installed (fresh on every reset)
            "source": "seed" if seed is not None else "os_entropy", "installed_at_reset": self.n_resets,
            **{f"{name}_seed_seq": {"entropy": str(g.bit_generator.seed_seq.entropy),
                                    "spawn_key": list(g.bit_generator.seed_seq.spawn_key)}
               for name, g in (("env", e.rng), ("sampler", e.placement_initializer.rng))},
        }
        self.cube_id, self.region_id = cubes[0], regions[0]
        e.reset()
        robot = e.robots[0]
        init_qpos = _l(e.sim.data.qpos[robot._ref_joint_pos_indexes])  # after init noise, before settling
        self.z_ref = float(e.table_offset[2] + e.cube.size[2])
        _, cx, cy = layout[self.cube_id]
        _, tx, ty = layout[self.region_id]
        e.sim.data.set_joint_qpos(e.cube.joints[0], np.array([cx, cy, self.z_ref, 1.0, 0.0, 0.0, 0.0]))
        e.set_target([tx, ty, self.z_ref])  # calls sim.forward()
        self._settle(5)
        # Realized initial conditions, so episodes can be compared and reproduced. robot_init_qpos is
        # taken right after reset (with init noise, before settling); poses are after the 5 settle steps.
        self.init = {
            "seed": seed, "seeded": seed is not None, "rng": rng_meta, "robot_init_qpos": init_qpos,
            "cube_half_size": _l(e.cube.size), "z_ref": float(e.table_offset[2] + e.cube.size[2]),
            "cube_pose_world": _l(self.obs["cube_pos"]), "cube_quat": _l(self.obs["cube_quat"]),
            "eef_pose_world": _l(self.obs["robot0_eef_pos"]),
        }
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


def _seed_generators(e, seed):
    if seed is not None:
        np.random.seed(seed)
        env_ss, sampler_ss = np.random.SeedSequence(seed).spawn(2)
        e.rng, e.placement_initializer.rng = np.random.default_rng(env_ss), np.random.default_rng(sampler_ss)
    else:
        e.rng, e.placement_initializer.rng = np.random.default_rng(), np.random.default_rng()
    e.seed = seed
    return {"source": "seed" if seed is not None else "os_entropy",
            **{f"{name}_seed_seq": {"entropy": str(g.bit_generator.seed_seq.entropy),
                                    "spawn_key": list(g.bit_generator.seed_seq.spawn_key)}
               for name, g in (("env", e.rng), ("sampler", e.placement_initializer.rng))}}


class TwoCubeBackend(Backend):
    """scene="two_cube" (F015): two cubes, two regions, barrier off the table. Shares Backend's low-level
    control helpers (_act, _settle, _retreat, _agent, _reach_failure); everything object-specific is here."""

    def __init__(self, render=False):
        self.scene = "two_cube"
        sampler = UniformRandomSampler(  # placements are overridden in reset(); kept for Lift's reset path
            name="ObjectSampler", x_range=[-0.08, 0.08], y_range=[-0.05, 0.02], rotation=None,
            ensure_object_boundary_in_range=True, ensure_valid_placement=True,
            reference_pos=[0, 0, 0.8], z_offset=0.01,
        )
        self.env = TwoCubeScene(
            barrier_pos=OFF_TABLE_BARRIER, robots="Panda", placement_initializer=sampler, has_renderer=render,
            has_offscreen_renderer=False, use_camera_obs=False, control_freq=10,
            horizon=100_000, ignore_done=True,
        )
        assert self.env.action_dim == 7, self.env.action_dim
        self.render = render
        self.init = None
        self.n_resets = 0
        self.perturb_at = {}

    def info(self):
        return {**super().info(), "scene": self.scene, "reach": list(REACH), "place_clearance": PLACE_CLEARANCE,
                "cubes": {k: {"half_size": self._half[k], "z_ref": self._zref[k]} for k in self.cube_ids},
                "regions": {k: list(v) for k, v in self.regions.items()}}

    def reset(self, layout, seed):
        cubes = [k for k, (kind, _, _) in layout.items() if kind == "cube"]
        regions = [k for k, (kind, _, _) in layout.items() if kind == "target_region"]
        if len(cubes) != 2 or len(regions) != 2 or len(layout) != 4:
            raise ValueError(f"two_cube scene needs exactly two cubes and two target_regions, got {layout}")
        for k, (_, x, y) in layout.items():
            if not (REACH[0] <= x <= REACH[1] and REACH[2] <= y <= REACH[3]):
                raise ValueError(f"{k} at ({x}, {y}) outside REACH {REACH}")
        (a, b), (r1, r2) = cubes, regions
        if np.hypot(layout[a][1] - layout[b][1], layout[a][2] - layout[b][2]) < PLACE_CLEARANCE:
            raise ValueError("initial cubes closer than PLACE_CLEARANCE")
        if abs(layout[r1][1] - layout[r2][1]) < 2 * REGION_HALF and abs(layout[r1][2] - layout[r2][2]) < 2 * REGION_HALF:
            raise ValueError("regions overlap")
        need = PLACE_CLEARANCE + PLACE_TOL  # a goal place lands within PLACE_TOL of a region centre
        if np.hypot(layout[r1][1] - layout[r2][1], layout[r1][2] - layout[r2][2]) < need:
            raise ValueError(f"region centres closer than {need} (finger clearance for goal places)")
        for c in cubes:
            for r in regions:
                inside = abs(layout[c][1] - layout[r][1]) < REGION_HALF and abs(layout[c][2] - layout[r][2]) < REGION_HALF
                if not inside and np.hypot(layout[c][1] - layout[r][1], layout[c][2] - layout[r][2]) < need:
                    raise ValueError(f"{c} is within {need} of region {r} centre (finger clearance)")
        e = self.env
        rng_meta = _seed_generators(e, seed)
        self.n_resets += 1
        rng_meta["installed_at_reset"] = self.n_resets
        # stable identities: a red-named id gets Lift's red cube, the other id the blue cube_b
        red = [c for c in cubes if "red" in c]
        first = red[0] if len(red) == 1 else cubes[0]
        self.cube_ids = [first, next(c for c in cubes if c != first)]
        self.region_ids = regions
        e.reset()
        robot = e.robots[0]
        init_qpos = _l(e.sim.data.qpos[robot._ref_joint_pos_indexes])
        self._obj = {self.cube_ids[0]: e.cube, self.cube_ids[1]: e.cube_b}
        self._body = {self.cube_ids[0]: e.cube_body_id, self.cube_ids[1]: e.cube_b_body_id}
        self._half = {k: float(o.size[2]) for k, o in self._obj.items()}
        table = float(e.table_offset[2])
        self._zref = {k: table + h for k, h in self._half.items()}
        self.z_ref = self._zref[self.cube_ids[0]]  # gripper height reference, as in the single-cube scene
        for k in self.cube_ids:
            _, x, y = layout[k]
            e.sim.data.set_joint_qpos(self._obj[k].joints[0], np.array([x, y, self._zref[k], 1.0, 0.0, 0.0, 0.0]))
            e.sim.data.set_joint_qvel(self._obj[k].joints[0], np.zeros(6))
        self.regions = {k: (float(layout[k][1]), float(layout[k][2])) for k in regions}
        e.set_target([*self.regions[r1], self._zref[self.cube_ids[0]]])  # moves target_marker; calls forward()
        bid = e.sim.model.body_name2id("region2_marker")
        e.sim.model.body_pos[bid] = np.array([*self.regions[r2], e.table_z + 0.002])
        e.sim.forward()
        self._settle(5)
        self.init = {"seed": seed, "seeded": seed is not None, "rng": rng_meta, "robot_init_qpos": init_qpos,
                     "cube_ids": self.cube_ids, "cube_half_size": dict(self._half), "z_ref": dict(self._zref),
                     "cube_pose_world": {k: _l(self._pos(k)) for k in self.cube_ids}, "regions": dict(self.regions),
                     "eef_pose_world": _l(self.obs["robot0_eef_pos"]), "barrier_pos": _l(e.barrier_pos)}
        self.t0 = float(e.sim.data.time)
        self.step = 0
        self.execute_calls = 0
        self.perturb_at = {}
        self.ctx = None
        return self.observe()

    def _pos(self, k):
        return self.env.sim.data.body_xpos[self._body[k]].copy()

    def _grasped_id(self):
        g = [k for k in self.cube_ids if self._grasped_obj(k)]
        return g[0] if g else None

    def _grasped_obj(self, k):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self._obj[k].contact_geoms))

    def _agent_cube(self, k):
        p = self._pos(k)
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2] - self._zref[k])}

    def observe(self):
        t = round(float(self.env.sim.data.time) - self.t0, 4)
        objects = {k: {"id": k, "kind": "cube", "pose": self._agent_cube(k), "visible": True, "confidence": 1.0,
                       "observed_at": t} for k in self.cube_ids}
        for k, (x, y) in self.regions.items():
            objects[k] = {"id": k, "kind": "target_region", "pose": {"x": x, "y": y, "z": 0.0}, "visible": True,
                          "confidence": 1.0, "observed_at": t}
        return {"t": t, "step": self.step, "objects": objects,
                "gripper": {"pose": self._agent(self.obs["robot0_eef_pos"]), "holding": self._grasped_id()}}

    def _region_of(self, k):
        p = self._pos(k)
        for rid, (x, y) in self.regions.items():
            if abs(p[0] - x) < REGION_HALF and abs(p[1] - y) < REGION_HALF:
                return rid
        return None

    def _tc_report(self, oid, termination, fsm_state, steps, ok, before, code=None, detail="", extra=None):
        poses = {k: _l(self._pos(k)) for k in self.cube_ids}
        moved = {k: float(np.hypot(*(np.array(poses[k][:2]) - np.array(before[k][:2]))))
                 for k in self.cube_ids if k != oid}
        info = {"control_steps": steps, "termination": termination, "fsm_state": fsm_state, "object_id": oid,
                "grasped": self._grasped_id(), "cube_poses_world": poses,
                "cube_pose": self._agent_cube(oid) if oid in self._obj else None,
                "eef_pose_world": _l(self.obs["robot0_eef_pos"]),
                "in_region": {k: self._region_of(k) for k in self.cube_ids},
                "uninvolved_displacement": moved, **(extra or {})}
        out = {"ok": ok, "info": info}
        if not ok:
            out["failure"] = {"code": code, "detail": detail}
        return out

    def schedule_perturb(self, schedule):
        """Test-only: {execute index: [cube id, x, y]} for the current episode (cleared on reset)."""
        self.perturb_at = {int(k): (v[0], float(v[1]), float(v[2])) for k, v in schedule.items()}
        return None

    def _perturb_conflict(self, oid, x, y):
        if oid not in self._obj:
            return f"unknown cube {oid!r}"
        if self._grasped_id() == oid:
            return f"{oid} is held"
        x0, x1, y0, y1 = REACH
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return f"({x}, {y}) outside REACH"
        for k in self.cube_ids:
            if k != oid:
                p = self._pos(k)
                if abs(p[0] - x) < CUBE_OVERLAP and abs(p[1] - y) < CUBE_OVERLAP:
                    return f"overlaps {k}"
        for rid, (rx, ry) in self.regions.items():
            if abs(x - rx) < REGION_HALF and abs(y - ry) < REGION_HALF:
                for k in self.cube_ids:
                    if k != oid and self._region_of(k) == rid:
                        return f"region {rid} already contains {k}"
        return None

    def execute(self, name, args):
        applied = None
        if (p := self.perturb_at.get(self.execute_calls)) is not None:
            oid, x, y = p
            if (why := self._perturb_conflict(oid, x, y)) is not None:  # refuse before changing anything
                raise ValueError(f"invalid test perturbation at execute {self.execute_calls}: {why}")
            j = self._obj[oid].joints[0]
            self.env.sim.data.set_joint_qpos(j, np.array([x, y, self._zref[oid], 1.0, 0.0, 0.0, 0.0]))
            self.env.sim.data.set_joint_qvel(j, np.zeros(6))
            self.env.sim.forward()
            self.obs = self.env._get_observations(force_update=True)
            applied = {"execute_index": self.execute_calls, "object_id": oid, "to": [x, y]}
        self.execute_calls += 1
        self.step += 1
        extra = {"test_perturbation": applied} if applied else {}
        if name == "pick":
            out = self._tc_pick(args["object_id"], args["x"], args["y"], extra)
        elif name == "place":
            out = self._tc_place(args["x"], args["y"], extra)
        else:
            out = {"ok": False, "failure": {"code": "UNKNOWN_TOOL", "detail": f"robosuite env has no skill {name!r}"}}
        if extra and "info" not in out:  # early refusals still record an applied perturbation
            out["info"] = dict(extra)
        return out

    def _transit(self, x, y):
        """Rise to retreat height, then move horizontally to above (x, y) so the approach descends vertically."""
        n = self._retreat()
        while True:
            eef = self.obs["robot0_eef_pos"]
            err = np.array([x - eef[0], y - eef[1], self.z_ref + RETREAT_Z - eef[2]])
            if np.hypot(err[0], err[1]) < TRANSIT_XY_TOL:
                return n, True
            if n >= TRANSIT_BUDGET:
                return n, False
            a = np.zeros(7, dtype=np.float32)
            a[:3] = np.clip(5.0 * err, -1.0, 1.0)
            a[6] = -1.0
            self._act(a)
            n += 1

    def _tc_pick(self, oid, x, y, extra):
        before = {k: _l(self._pos(k)) for k in self.cube_ids}
        if oid not in self._obj:
            return {"ok": False, "failure": {"code": "UNKNOWN_OBJECT", "detail": f"{oid!r} is not graspable"}}
        if self._grasped_id() is not None:
            return {"ok": False, "failure": {"code": "ALREADY_HOLDING", "detail": f"already holding {self._grasped_id()}"}}
        if (f := self._reach_failure(x, y)) is not None:
            return {"ok": False, "failure": f}
        steps, converged = self._transit(x, y)
        extra = {**extra, "approach_start_eef_world": _l(self.obs["robot0_eef_pos"]), "transit_converged": converged}
        if not converged:  # never descend at an angle next to another cube
            return self._tc_report(oid, "transit_budget", "TRANSIT", steps, False, before, "POSTCONDITION_FAILED",
                                   f"could not reach retreat height above ({x:.3f}, {y:.3f}); pick not attempted", extra)
        zr = self._zref[oid]
        target = np.array([x, y, zr])
        commanded = np.array([x, y, zr])
        ctx = {"grasped": False, "grasp_offset": None, "grasp_count": 0, "route": None, "transport_stage": 0}
        state, fsm_steps, grasp_steps, established, termination = "APPROACH", 0, 0, False, "step_budget"
        while fsm_steps < PICK_BUDGET:
            grasped = self._grasped_obj(oid)
            if established and not grasped:
                termination = "grasp_lost"
                break
            ctx["grasped"] = grasped
            obs = dict(self.obs)
            obs["cube_pos"] = commanded if state in ("APPROACH", "DESCEND") else self._pos(oid)
            prev = state
            action, state = constrained_pick_place_expert_action(obs, state, target, ctx)
            if prev == "GRASP" and state == "LIFT":
                established = True
            self._act(action)
            fsm_steps += 1
            if state == "GRASP":
                grasp_steps += 1
                if grasp_steps > GRASP_PATIENCE:
                    termination = "no_grasp_contact"
                    break
            if state == "TRANSPORT":
                termination = "fsm_lifted"
                break
        steps += fsm_steps
        self.ctx = ctx
        cube = self._pos(oid)
        grasped = self._grasped_obj(oid)
        lifted = cube[2] - zr >= LIFTED_Z
        if termination == "fsm_lifted" and grasped and lifted:
            return self._tc_report(oid, termination, state, steps, True, before, extra=extra)
        steps += self._retreat()
        cube = self._pos(oid)
        err = float(np.hypot(cube[0] - x, cube[1] - y))
        if established and not grasped:
            return self._tc_report(oid, termination, state, steps, False, before, "GRASP_SLIP",
                                   f"{oid} slipped out of the gripper during {state} (established grasp lost)", extra)
        if not established and err > GRASP_TOL:
            return self._tc_report(oid, termination, state, steps, False, before, "OBJECT_NOT_AT_POSE",
                                   f"closed gripper at ({x:.3f}, {y:.3f}) but {oid} is {err:.3f} m away", extra)
        return self._tc_report(oid, termination, state, steps, False, before, "POSTCONDITION_FAILED",
                               f"pick ended ({termination}, fsm={state}) with grasped={grasped}, "
                               f"cube z={cube[2] - zr:.3f} (need >= {LIFTED_Z})", extra)

    def _tc_place(self, x, y, extra):
        before = {k: _l(self._pos(k)) for k in self.cube_ids}
        oid = self._grasped_id()
        if oid is None or self.ctx is None or self.ctx.get("grasp_offset") is None:
            return {"ok": False, "failure": {"code": "NOT_HOLDING", "detail": "gripper is empty"}}
        if (f := self._reach_failure(x, y)) is not None:
            return {"ok": False, "failure": f}
        for k in self.cube_ids:  # open-finger clearance (refused without motion)
            if k != oid:
                p = self._pos(k)
                d = float(np.hypot(p[0] - x, p[1] - y))
                if d < PLACE_CLEARANCE:
                    return self._tc_report(oid, "refused", "TRANSPORT", 0, False, before, "DESTINATION_OCCUPIED",
                                           f"place at ({x:.3f}, {y:.3f}) is {d:.3f} m from {k}; finger clearance "
                                           f"needs >= {PLACE_CLEARANCE}", extra)
        ctx = self.ctx
        ctx["route"] = "LEFT"
        ctx["transport_stage"] = 2  # barrier is off the table: never crossed
        zr = self._zref[oid]
        target = np.array([x, y, zr])
        state, steps, termination = "TRANSPORT", 0, "step_budget"
        while steps < PLACE_BUDGET:
            grasped = self._grasped_obj(oid)
            if state in ("TRANSPORT", "LOWER") and not grasped:
                termination = "grasp_lost"
                break
            ctx["grasped"] = grasped
            obs = dict(self.obs)
            obs["cube_pos"] = self._pos(oid)
            action, state = constrained_pick_place_expert_action(obs, state, target, ctx)
            self._act(action)
            steps += 1
            if state == "DONE":
                termination = "fsm_released"
                break
        steps += self._retreat()
        self._settle(5)
        steps += 5
        self.ctx = None
        cube = self._pos(oid)
        grasped = self._grasped_obj(oid)
        err = float(np.hypot(cube[0] - x, cube[1] - y))
        z = float(cube[2] - zr)
        if termination == "grasp_lost":
            return self._tc_report(oid, termination, state, steps, False, before, "GRASP_SLIP",
                                   f"cube dropped during {state}; now at ({cube[0]:.3f}, {cube[1]:.3f}, z={z:.3f})", extra)
        if not grasped and err <= PLACE_TOL and abs(z) <= REST_Z_TOL:
            return self._tc_report(oid, termination, state, steps, True, before, extra=extra)
        return self._tc_report(oid, termination, state, steps, False, before, "POSTCONDITION_FAILED",
                               f"place ended ({termination}, fsm={state}) with grasped={grasped}, cube "
                               f"{err:.3f} m from ({x:.3f}, {y:.3f}) at z={z:.3f}", extra)


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
                scene = req.get("scene", "constrained")
                if scene not in ("constrained", "two_cube"):
                    raise ValueError(f"unknown scene {scene!r}")
                backend = (TwoCubeBackend if scene == "two_cube" else Backend)(render=req.get("render", False))
                result = None
            elif op == "schedule_perturb":  # test-only (two_cube scene)
                if not isinstance(backend, TwoCubeBackend):
                    raise ValueError("schedule_perturb is only supported by the two_cube scene")
                result = backend.schedule_perturb(req["schedule"])
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
