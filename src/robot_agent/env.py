"""Env interface and a deterministic kinematic mock tabletop.

A future robosuite backend implements the same three methods of `RobotEnv`.
"""

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from robot_agent.goals import TaskSpec
from robot_agent.world import (
    Failure,
    FailureCode,
    GripperObs,
    ObjectObs,
    Pose,
    SkillOutcome,
    WorldState,
)

GRASP_TOL = 0.02  # max xy error between commanded pick point and object, metres
CARRY_Z = 0.10
SKILL_DURATION = 0.5  # env seconds consumed per executed skill


class RobotEnv(Protocol):
    def reset(self, task: TaskSpec, seed: int | None = None) -> WorldState: ...
    def observe(self) -> WorldState: ...
    def execute(self, name: str, args: BaseModel) -> SkillOutcome: ...


@dataclass
class Faults:
    """Deterministic, scheduled fault injection for the mock."""

    # skill name -> failure returned on the first call of that skill (consumed once)
    fail_once: dict[str, FailureCode] = field(default_factory=dict)
    # execute-call index -> (object id, x, y): object is moved just before that call runs
    perturb_at: dict[int, tuple[str, float, float]] = field(default_factory=dict)
    # first place reports ok, but the object actually lands at this xy (consumed once)
    silent_drop_on_place: tuple[float, float] | None = None


class MockTabletopEnv:
    def __init__(self, faults: Faults | None = None):
        self.faults = faults or Faults()
        self.execute_calls = 0  # every call to execute, including failed ones

    def reset(self, task: TaskSpec, seed: int | None = None) -> WorldState:
        self.seed = seed
        self.t = 0.0
        self.step = 0
        self.execute_calls = 0
        self._fail_once = dict(self.faults.fail_once)
        self._silent_drop = self.faults.silent_drop_on_place
        self.poses = {oid: Pose(x=x, y=y) for oid, (_, x, y) in task.layout.items()}
        self.kinds = {oid: kind for oid, (kind, _, _) in task.layout.items()}
        self.gripper = Pose(x=0.0, y=0.0, z=0.3)
        self.holding: str | None = None
        return self.observe()

    def observe(self) -> WorldState:
        # Perfect perception in the mock; a real backend fills visibility/confidence from sensing.
        objects = {
            oid: ObjectObs(
                id=oid,
                kind=self.kinds[oid],
                pose=pose.model_copy(),
                visible=True,
                confidence=1.0,
                observed_at=self.t,
            )
            for oid, pose in self.poses.items()
        }
        return WorldState(
            t=self.t,
            step=self.step,
            objects=objects,
            gripper=GripperObs(pose=self.gripper.model_copy(), holding=self.holding),
        )

    def execute(self, name: str, args: BaseModel) -> SkillOutcome:
        if (p := self.faults.perturb_at.get(self.execute_calls)) is not None:
            oid, x, y = p
            self.poses[oid] = Pose(x=x, y=y)
        self.execute_calls += 1
        self.step += 1
        self.t += SKILL_DURATION
        if name == "pick":
            return self._pick(args.object_id, args.x, args.y)
        if name == "place":
            return self._place(args.x, args.y)
        return _fail(FailureCode.UNKNOWN_TOOL, f"mock env has no skill {name!r}")

    def _pick(self, oid: str, x: float, y: float) -> SkillOutcome:
        if self.holding is not None:
            return _fail(FailureCode.ALREADY_HOLDING, f"already holding {self.holding}")
        if oid not in self.poses:
            return _fail(FailureCode.UNKNOWN_OBJECT, f"no object {oid!r}")
        self.gripper = Pose(x=x, y=y, z=0.0)
        err = self.poses[oid].dist_xy(x, y)
        if err > GRASP_TOL:
            self.gripper.z = CARRY_Z
            return _fail(
                FailureCode.OBJECT_NOT_AT_POSE,
                f"closed gripper at ({x:.3f}, {y:.3f}) but {oid} is {err:.3f} m away",
            )
        if self._fail_once.pop("pick", None) is not None:
            self.gripper.z = CARRY_Z
            return _fail(FailureCode.GRASP_SLIP, f"{oid} slipped out of the gripper")
        self.holding = oid
        self.gripper.z = CARRY_Z
        self.poses[oid] = Pose(x=x, y=y, z=CARRY_Z)
        return SkillOutcome(ok=True)

    def _place(self, x: float, y: float) -> SkillOutcome:
        if self.holding is None:
            return _fail(FailureCode.NOT_HOLDING, "gripper is empty")
        oid, self.holding = self.holding, None
        self.gripper = Pose(x=x, y=y, z=CARRY_Z)
        if self._silent_drop is not None:
            (x, y), self._silent_drop = self._silent_drop, None
        self.poses[oid] = Pose(x=x, y=y)
        return SkillOutcome(ok=True)  # reports ok even after a silent drop


def _fail(code: FailureCode, detail: str) -> SkillOutcome:
    return SkillOutcome(ok=False, failure=Failure(code=code, detail=detail))
