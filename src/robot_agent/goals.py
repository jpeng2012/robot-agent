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


class TaskSpec(BaseModel):
    name: str
    instruction: str
    layout: dict[str, tuple[str, float, float]]  # object id -> (kind, x, y)
    goal: list[On]


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
