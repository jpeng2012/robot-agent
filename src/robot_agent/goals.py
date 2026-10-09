"""Task specs and task-level goal verification (independent of skill postconditions)."""

from pydantic import BaseModel

from robot_agent.world import WorldState


class On(BaseModel):
    """`obj` rests on `support`: within `tol` of its xy and not held by the gripper."""

    obj: str
    support: str
    tol: float = 0.05

    def holds(self, s: WorldState) -> bool:
        if self.obj not in s.objects or self.support not in s.objects:
            return False
        if s.gripper.holding == self.obj:
            return False
        sup = s.objects[self.support].pose
        return s.objects[self.obj].pose.dist_xy(sup.x, sup.y) <= self.tol

    def __str__(self) -> str:
        return f"on({self.obj}, {self.support})"


class InRegion(BaseModel):
    """`obj` rests on the table inside the axis-aligned xy box around `region`, and is not held.

    `region` is a flat target area on the table (e.g. a visual marker), not a support surface,
    so z is checked against table level (z = 0) rather than against the region's height.
    """

    obj: str
    region: str
    half_x: float
    half_y: float
    z_tol: float = 0.025

    def holds(self, s: WorldState) -> bool:
        if self.obj not in s.objects or self.region not in s.objects:
            return False
        if s.gripper.holding == self.obj:
            return False
        o, r = s.objects[self.obj].pose, s.objects[self.region].pose
        return abs(o.x - r.x) < self.half_x and abs(o.y - r.y) < self.half_y and abs(o.z) <= self.z_tol

    def __str__(self) -> str:
        return f"in_region({self.obj}, {self.region})"


class TaskSpec(BaseModel):
    name: str
    instruction: str
    layout: dict[str, tuple[str, float, float]]  # object id -> (kind, x, y)
    goal: list[On | InRegion]


def check_goal(task: TaskSpec, state: WorldState) -> list[str]:
    """Return the unmet goal predicates; an empty list means the task is achieved."""
    return [str(p) for p in task.goal if not p.holds(state)]


def pick_place_task() -> TaskSpec:
    return TaskSpec(
        name="pick_place",
        instruction="Put the red cube on the plate.",
        layout={"cube_red": ("cube", 0.10, -0.20), "plate": ("plate", -0.20, 0.15)},
        goal=[On(obj="cube_red", support="plate")],
    )


def region_pick_place_task() -> TaskSpec:
    """Robosuite ConstrainedPickPlace: cube in front of the barrier, green target region behind it."""
    return TaskSpec(
        name="region_pick_place",
        instruction="Put the red cube inside the green target region.",
        layout={"cube_red": ("cube", 0.0, -0.02), "target": ("target_region", 0.05, 0.30)},
        goal=[InRegion(obj="cube_red", region="target", half_x=0.045, half_y=0.045)],
    )
