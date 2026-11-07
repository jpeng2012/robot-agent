# Robot-Agent — Development Instructions

## Project
LLM-driven robot agent using typed skills and
a backend-independent robot execution interface.

## Session Startup
- Read this file and claude-progress.md.
- Inspect feature_list.json.
- Review git status and recent commits.
- Run baseline verification before modifying code.
- Report existing failures.

## Working Rules
- Work on one feature at a time.
- Inspect existing interfaces before proposing changes.
- Prefer minimal changes over new frameworks.
- Preserve existing behavior unless explicitly changed.
- Do not hardcode solutions to pass specific tests.
- Do not weaken tests to claim success.
- Ask for design approval before major architectural changes.

## Robotics Invariants
- Controller completion is not proof of task success.
- Report failure and uncertain outcomes explicitly.
- Never bypass safety or execution limits.
- Validate state freshness before physical actions.
- Maintain compatibility with existing backends.

## Verification
- Run relevant unit tests.
- Run mock-backend integration tests.
- Run simulator tests when available.
- Report actual results, skipped tests, and limitations.
- Only mark a feature complete with recorded evidence.

## Session Handoff
- Summarize changes and design decisions.
- Record verification commands and outcomes.
- Document remaining failures.
- Update progress and feature state.