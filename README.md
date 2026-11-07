# robot-agent

LLM-driven robot agent using typed skills and a backend-independent robot execution interface.

## Overview

A minimal closed-loop robotics system with tool calling via LLM and deterministic kinematic mock for tabletop manipulation. The agent observes the world, plans actions using an LLM, executes skills, and verifies goal achievement.

**Key Features:**
- Protocol-based backend abstraction (mock, simulator, real robot)
- Bounded retry logic for transient failures
- Independent goal verification (controller success ≠ task success)
- Full JSONL trace logging for analysis
- Multiple LLM backends (Ollama, rule-based policy, scripted)

## Quick Start

### Installation

```bash
# Install dependencies using uv
uv sync

# Or manually create venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verification

```bash
# Run initialization script (checks environment + runs tests)
./init.sh

# Or run tests directly
.venv/bin/python -m pytest tests/ -v
# Expected: 39 passed, 5 skipped in ~0.2s
```

### Basic Usage

```bash
# Run with deterministic mock backend and rule-based policy
robot-agent --backend policy --scenario happy

# Run with fault injection
robot-agent --backend scripted --scenario slip

# Run with LLM (requires Ollama server)
robot-agent --backend ollama --model qwen3:8b --repeat 3

# Run with robosuite physics simulator (requires robosuite)
robot-agent --backend policy --env robosuite --render
```

## Architecture

### Core Components

**RobotEnv Protocol** ([env.py](src/robot_agent/env.py))
- `reset(task, seed)` - Initialize environment
- `observe()` - Get current world state
- `execute(name, args)` - Execute a skill

**Skills** ([skills.py](src/robot_agent/skills.py))
- `observe` - Get world state
- `pick(x, y)` - Grasp object at location
- `place(x, y)` - Release held object
- `finish()` - Declare task complete

**Agent Loop** ([loop.py](src/robot_agent/loop.py))
- Observe → Plan (LLM) → Act → Verify cycle
- Retry budget for transient failures (GRASP_SLIP)
- Goal checking after finish skill
- Full trace logging to JSONL

### World State

Typed state representation using Pydantic ([world.py](src/robot_agent/world.py)):

```python
@dataclass
class WorldState:
    t: float              # Timestamp
    step: int             # Step counter
    objects: dict[str, ObjectObs]  # Visible objects
    gripper: GripperObs   # Gripper state

@dataclass
class ObjectObs:
    pose: Pose            # 3D position
    visible: bool
    confidence: float

@dataclass
class GripperObs:
    pose: Pose
    holding: str | None   # Object ID or None
```

## Backends

### 1. Mock Tabletop Environment

**Implementation:** `MockTabletopEnv` in [env.py](src/robot_agent/env.py)

- Deterministic 2D kinematics
- Perfect perception (no noise)
- Workspace: x ∈ [-0.5, 0.5], y ∈ [-0.5, 0.5]
- Fault injection support:
  - `fail_once`: Single-use failure injection
  - `perturb_at`: Move objects before execution
  - `silent_drop_on_place`: Object lands elsewhere

**Testing:**
```bash
pytest tests/test_env.py -v
# 3/3 tests: observation, pick/place, perturbation
```

### 2. Robosuite Physics Simulator

**Implementation:** `RobosuiteEnv` in [robosuite_env.py](src/robot_agent/robosuite_env.py)

- Subprocess bridge to Python 3.10 worker
- Panda arm with OSC controller (10 Hz)
- ConstrainedPickPlace scene from robot-policy-learning
- Region-based goal checking
- Coordinate alignment: z=0 at resting height

**Setup:**
```bash
# Requires Python 3.10 and MuJoCo
conda create -n robosuite python=3.10
conda activate robosuite
pip install robosuite
```

**Testing:**
```bash
pytest tests/test_robosuite_env.py -v
# 7/11 pass, 4 skip (require robosuite installation)
```

### 3. LLM Backends

**OllamaLLMClient** ([llm.py](src/robot_agent/llm.py))
- OpenAI-compatible HTTP interface
- Qwen3 thinking mode (optional, disabled by default)
- Temperature control (default: 0.0)
- Timeout handling

**Setup:**
```bash
# Install and start Ollama server
ollama serve &
ollama pull qwen3:8b

# Run agent with LLM
robot-agent --backend ollama --model qwen3:8b
```

**PickPlacePolicyLLM** ([llm.py](src/robot_agent/llm.py))
- Rule-based deterministic policy
- Always observes first
- Grounds pick on latest pose

**ScriptedLLM** ([llm.py](src/robot_agent/llm.py))
- Fixed tool-call sequences
- Used for testing specific scenarios

## Testing

### Test Suite Overview

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_env.py` | 3/3 | Mock environment behavior |
| `test_loop.py` | 20/20 | Agent control loop, retry logic, validation |
| `test_skills.py` | 3/3 | Schema generation, validation, verification |
| `test_ollama_client.py` | 6/7 (1 skip) | LLM integration |
| `test_robosuite_env.py` | 7/11 (4 skip) | Physics simulator |

**Total:** 39 passed, 5 skipped

### Running Tests

```bash
# All tests
.venv/bin/python -m pytest tests/ -v

# Specific test file
pytest tests/test_loop.py -v

# Exclude skipped tests
pytest tests/ -v -k 'not requires_server and not requires_robosuite'

# Single test
pytest tests/test_loop.py::test_failed_pick_recovers_by_bounded_retry -v
```

### Test Scenarios

**Fault Injection Scenarios:**
- `happy` - Nominal execution
- `slip` - Grasp failure with bounded retry
- `perturb` - Object moved after planning
- `silent_drop` - Object lands elsewhere than commanded
- `premature_finish` - Agent declares success without achieving goal

## Trace Data

Episodes are logged as JSONL to `traces/` (excluded from git):

```json
{
  "task": {"goal": [{"On": {"object": "cube_red", "support": "plate"}}]},
  "seed": 42,
  "step": 0,
  "state": {"t": 0.0, "step": 0, "objects": {...}, "gripper": {...}},
  "tool_call": {"name": "observe", "args": {}},
  "result": {"code": "OK", "message": ""},
  "goal_met": false
}
```

**Trace Collections:**
- `traces/baseline/` - 8 baseline episodes
- `traces/qwen/` - 44 Qwen3 LLM episodes
- `traces/qwen2/` - 31 Qwen2 LLM episodes

## CLI Options

```bash
robot-agent [OPTIONS]

Options:
  --backend [scripted|policy|ollama]  # Agent backend (default: scripted)
  --scenario [happy|slip|perturb|silent_drop|premature_finish]  # Fault injection
  --env [mock|robosuite]              # Environment backend (default: mock)
  --repeat N                          # Run N episodes with seed offset
  --model MODEL                       # Ollama model name (default: qwen3:8b)
  --base-url URL                      # Ollama base URL (default: http://localhost:11434/v1)
  --think                             # Enable Qwen3 thinking mode
  --render                            # Render robosuite GUI
  --trace-dir DIR                     # Trace output directory (default: traces)
```

## Development Workflow

See [CLAUDE.md](CLAUDE.md) for AI-assisted development instructions.

### Session Startup

1. Review [CLAUDE.md](CLAUDE.md) and [claude-progress.md](claude-progress.md)
2. Check [feature_list.json](feature_list.json) for task scope
3. Run `./init.sh` for baseline verification
4. Work on one feature at a time

### Verification Requirements

- Run relevant unit tests
- Run mock-backend integration tests
- Run simulator tests when available
- Report actual results, skipped tests, and limitations
- Only mark features complete with recorded evidence

## Project Structure

```
robot-agent/
├── src/robot_agent/          # Source code (1,269 lines)
│   ├── cli.py               # CLI and episode runner
│   ├── env.py               # RobotEnv protocol, MockTabletopEnv
│   ├── goals.py             # Task specs and goal checking
│   ├── llm.py               # LLM providers
│   ├── loop.py              # Agent control loop
│   ├── robosuite_env.py     # Robosuite backend
│   ├── robosuite_worker.py  # Robosuite worker process
│   ├── skills.py            # Skill definitions
│   ├── trace.py             # JSONL logging
│   └── world.py             # World state types
├── tests/                   # Test suite (44 tests)
├── traces/                  # Episode logs (excluded from git)
├── CLAUDE.md                # Development instructions
├── claude-progress.md       # Session history
├── feature_list.json        # Task scope tracking
├── init.sh                  # Baseline verification script
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Design Principles

1. **Protocol-based backends** - Swap mock/simulator/real robot via RobotEnv
2. **Provider-neutral LLM** - Works with Ollama, policies, or scripted agents
3. **Bounded failure recovery** - Retry only transient failures (GRASP_SLIP)
4. **Independent goal verification** - Separate from skill postconditions
5. **Full trace logging** - Every decision, failure, outcome recorded
6. **State validation** - Pre-conditions check freshness, post-conditions verify outcomes
7. **Deterministic testing** - Mock environment enables reproducible scenarios

## Robotics Invariants

From [CLAUDE.md](CLAUDE.md):

- Controller completion is not proof of task success
- Report failure and uncertain outcomes explicitly
- Never bypass safety or execution limits
- Validate state freshness before physical actions
- Maintain compatibility with existing backends

## License

See project repository for license information.

## Contributing

Follow the development workflow in [CLAUDE.md](CLAUDE.md). All features require:
- Acceptance criteria in [feature_list.json](feature_list.json)
- Passing tests with recorded evidence
- Updated session history in [claude-progress.md](claude-progress.md)
