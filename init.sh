#!/usr/bin/env bash
set -euo pipefail

echo "🤖 Robot-Agent Development Environment Check"
echo "============================================="
echo ""

# 1. Environment verification
echo "📋 Environment:"
if [ ! -f ".python-version" ]; then
    echo "  ❌ .python-version missing"
    exit 1
fi
EXPECTED_PY=$(cat .python-version)
echo "  ✅ Python version: $EXPECTED_PY"

if [ ! -d ".venv" ]; then
    echo "  ❌ Virtual environment not found (.venv)"
    echo "     Run: uv sync"
    exit 1
fi
echo "  ✅ Virtual environment exists"

# Always use project's .venv to avoid environment conflicts
PYTEST_CMD=".venv/bin/python -m pytest"
echo "  ✅ Using project virtual environment"

# 2. Git status check
echo ""
echo "📊 Git Status:"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "  Branch: $BRANCH"
LAST_COMMIT=$(git log -1 --oneline 2>/dev/null || echo "no commits")
echo "  Last commit: $LAST_COMMIT"

UNTRACKED=$(git ls-files --others --exclude-standard | wc -l)
MODIFIED=$(git diff --name-only | wc -l)
STAGED=$(git diff --cached --name-only | wc -l)

if [ "$UNTRACKED" -gt 0 ] || [ "$MODIFIED" -gt 0 ] || [ "$STAGED" -gt 0 ]; then
    echo "  ⚠️  Working tree has changes:"
    [ "$UNTRACKED" -gt 0 ] && echo "      Untracked: $UNTRACKED files"
    [ "$MODIFIED" -gt 0 ] && echo "      Modified: $MODIFIED files"
    [ "$STAGED" -gt 0 ] && echo "      Staged: $STAGED files"
else
    echo "  ✅ Working tree clean"
fi

# 3. Baseline verification
echo ""
echo "🧪 Running Baseline Tests:"
echo "  Command: $PYTEST_CMD tests/ -v"
echo ""

$PYTEST_CMD tests/ -v

echo ""
echo "✅ Initialization complete. Review CLAUDE.md and claude-progress.md before starting work."
