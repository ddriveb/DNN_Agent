#!/bin/bash
set -e

# Resolve script directory (works with symlinks and regardless of caller CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/../tests"

if [ ! -d "$TESTS_DIR" ]; then
    echo "Error: cannot locate tests/ directory at $TESTS_DIR"
    exit 1
fi

# Resolve python interpreter
# Priority: PYTHON env var > python3 > python
if [ -n "${PYTHON:-}" ]; then
    PYTHON_CMD="$PYTHON"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: no python interpreter found (tried python3, python)"
    echo "Set PYTHON env var to override, e.g.: PYTHON=/path/to/python bash $0"
    exit 1
fi

echo "=== Running SA-HMARL Unit Tests ==="
echo "Tests dir: $TESTS_DIR"
echo "Python:    $PYTHON_CMD"
echo ""

for f in "$TESTS_DIR"/test_*.py; do
    echo "=== $(basename "$f") ==="
    "$PYTHON_CMD" "$f"
done

echo ""
echo "=== All tests passed ==="
