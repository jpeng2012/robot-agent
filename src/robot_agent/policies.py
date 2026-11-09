"""Deterministic multi-object policies for region goals (F011b-1).

Both policies are LLMClient implementations that act only on the latest world_state delivered in a
tool result (llm.latest_state). They keep no state between steps and emit at most one tool call per
turn, so every pick/place is grounded on the state returned by the previous call (F010).

NaivePolicy: works through the subgoals in the supplied order and ignores occupied destinations.
DependencyAwarePolicy: if the current subgoal's destination is occupied, it first moves the blocker to
the blocker's own free goal region or to a staging point outside every region, then resumes.

Text-only stops carry a structured code in meta["code"] (STOP_CODES); the text is unchanged. Staging is
parameterised by workspace bounds (default WORKSPACE) and shared with the symbolic planner (planner.py).
"""

from robot_agent.goals import InRegion
from robot_agent.llm import AssistantTurn, call, latest_state
from robot_agent.world import CUBE_HALF, REGION_HALF, WORKSPACE, WorldState, placement_conflict

STAGING_STEP = 0.05
STAGING_MARGIN = 0.05  # keep staging points this far inside the WORKSPACE bounds
STAGING_CLEARANCE = REGION_HALF + CUBE_HALF  # a staged cube's footprint stays outside every region
STOP_CODES = ("NO_SUBGOAL_FOR_HELD", "NO_STAGING", "BLOCKER_NOT_FOUND", "CONFLICTING_GOALS")


def _xy(s: WorldState, oid: str) -> tuple[float, float]:
    p = s.objects[oid].pose
    return p.x, p.y


def _inside(s: WorldState, region: str, x: float, y: float) -> bool:
    rx, ry = _xy(s, region)
    return abs(x - rx) < REGION_HALF and abs(y - ry) < REGION_HALF


def _staging_points(bounds: tuple[float, float, float, float] = WORKSPACE) -> list[tuple[float, float]]:
    """Grid STAGING_STEP apart, STAGING_MARGIN inside bounds (x0, x1, y0, y1); empty if bounds are too small."""
    x0, x1, y0, y1 = bounds
    nx = int(round((x1 - x0 - 2 * STAGING_MARGIN) / STAGING_STEP))
    ny = int(round((y1 - y0 - 2 * STAGING_MARGIN) / STAGING_STEP))
    if nx < 0 or ny < 0:
        return []
    xs = [round(x0 + STAGING_MARGIN + STAGING_STEP * i, 3) for i in range(nx + 1)]
    ys = [round(y0 + STAGING_MARGIN + STAGING_STEP * i, 3) for i in range(ny + 1)]
    return [(x, y) for x in xs for y in ys]


def _staging(s: WorldState, held: str,
             bounds: tuple[float, float, float, float] = WORKSPACE) -> tuple[float, float] | None:
    """Nearest free grid point (to the held cube's pre-pick xy) whose footprint is outside every region."""
    hx, hy = _xy(s, held)
    regions = [_xy(s, rid) for rid, o in s.objects.items() if o.kind == "target_region"]
    ok = [(x, y) for x, y in _staging_points(bounds)
          if all(abs(x - rx) >= STAGING_CLEARANCE or abs(y - ry) >= STAGING_CLEARANCE for rx, ry in regions)
          and placement_conflict(s, held, x, y) is None]
    return min(ok, key=lambda p: (round(((p[0] - hx) ** 2 + (p[1] - hy) ** 2) ** 0.5, 6), p[0], p[1]), default=None)


def _blocker(s: WorldState, g: InRegion) -> str | None:
    """A cube other than g.obj that occupies g's region or overlaps its centre (first by id)."""
    dx, dy = _xy(s, g.region)
    for oid in sorted(s.objects):
        o = s.objects[oid]
        if o.kind != "cube" or oid == g.obj or s.gripper.holding == oid:
            continue
        if _inside(s, g.region, o.pose.x, o.pose.y) or (
                abs(o.pose.x - dx) < 2 * CUBE_HALF and abs(o.pose.y - dy) < 2 * CUBE_HALF):
            return oid
    return None


def _text(msg: str, code: str, **meta) -> AssistantTurn:
    if code not in STOP_CODES:
        raise ValueError(f"unknown stop code {code}")
    return AssistantTurn(text=msg, meta={"mode": "stop", "code": code, **meta})


def _act(name: str, mode: str, **args) -> AssistantTurn:
    return AssistantTurn(tool_calls=[call(name, **args)], meta={"mode": mode})


def decide(s: WorldState | None, subgoals: list[InRegion], aware: bool,
           bounds: tuple[float, float, float, float] = WORKSPACE) -> AssistantTurn:
    if s is None:
        return _act("observe", "observe")
    if all(g.holds(s) for g in subgoals):
        return _act("finish", "finish", summary="all subgoals hold in the latest world state")
    goal_of = {g.obj: g for g in subgoals}
    held = s.gripper.holding
    if held is not None:
        g = goal_of.get(held)
        if not aware:
            if g is None:
                return _text(f"holding {held}, which has no subgoal", "NO_SUBGOAL_FOR_HELD")
            x, y = _xy(s, g.region)
            return _act("place", "goal", x=x, y=y)
        if g is not None and placement_conflict(s, held, *_xy(s, g.region)) is None:
            x, y = _xy(s, g.region)
            return _act("place", "goal", x=x, y=y)
        spot = _staging(s, held, bounds)
        if spot is None:
            return _text(f"no valid staging location for {held}", "NO_STAGING")
        return _act("place", "stage", x=spot[0], y=spot[1])
    g = next(g for g in subgoals if not g.holds(s))
    if aware and placement_conflict(s, g.obj, *_xy(s, g.region)) is not None:
        b = _blocker(s, g)
        if b is None:
            return _text(f"{g} is blocked but no blocking cube was found", "BLOCKER_NOT_FOUND")
        if b in goal_of and goal_of[b].holds(s):
            return _text(f"conflicting goals: {b} already satisfies its subgoal but blocks {g}", "CONFLICTING_GOALS")
        x, y = _xy(s, b)
        turn = _act("pick", "clear_blocker", object_id=b, x=x, y=y)
        turn.meta.update(subgoal=str(g), blocker=b)
        return turn
    x, y = _xy(s, g.obj)
    return _act("pick", "goal", object_id=g.obj, x=x, y=y)


class _SubgoalPolicy:
    aware: bool

    def __init__(self, subgoals: list[InRegion]):
        subgoals = list(subgoals)
        if not all(isinstance(g, InRegion) for g in subgoals):
            raise ValueError("only InRegion subgoals are supported")
        if len({g.region for g in subgoals}) != len(subgoals) or len({g.obj for g in subgoals}) != len(subgoals):
            raise ValueError("each region and each object may appear in at most one subgoal")
        self.subgoals = subgoals

    def step(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        return decide(latest_state(messages), self.subgoals, self.aware)


class NaivePolicy(_SubgoalPolicy):
    """Follows the subgoals in the supplied order; never resolves an occupied destination."""
    aware = False


class DependencyAwarePolicy(_SubgoalPolicy):
    """Clears a blocked destination (blocker to its own free goal, else to staging), then resumes."""
    aware = True
