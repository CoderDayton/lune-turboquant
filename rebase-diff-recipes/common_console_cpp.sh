#!/usr/bin/env bash
# Auto-generated merge recipe for: common/console.cpp
set -euo pipefail
cd "/mnt/storage/rebase-tq"

# 1. See the upstream-only changes after merge-base.
echo "=== upstream-only changes in common/console.cpp ==="
git diff --unified=3 "1fd6dfe9f3d4b69cce101d832339fbda2d14b056..upstream/master" -- "common/console.cpp" | sed -n '1,220p' || true

# 2. See the TurboQuant-only changes after merge-base.
echo "=== TurboQuant-only changes in common/console.cpp ==="
git diff --unified=3 "1fd6dfe9f3d4b69cce101d832339fbda2d14b056..HEAD" -- "common/console.cpp" | sed -n '1,220p' || true

# 3. Produce a three-way merge scaffold if conflict occurs after cherry-pick.
echo "=== three-way merge scaffold ==="
echo '# After a cherry-pick conflict on this file, run:'
echo '#   git checkout --ours  <file>   # or --theirs, depending on target'
echo '#   then apply only the TurboQuant-specific hunks below:'
