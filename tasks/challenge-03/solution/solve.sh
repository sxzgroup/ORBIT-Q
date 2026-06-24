#!/bin/bash
set -euo pipefail

problem_id="$(cat /solution/problem_id.txt)"
cp "/solution/solution_${problem_id}.py" "/root/solution_${problem_id}.py"
