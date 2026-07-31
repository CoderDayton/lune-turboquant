#!/usr/bin/env bash
# Auto-generated merge recipe for: ggml/include/ggml.h
set -euo pipefail
cd "/mnt/storage/rebase-tq"

# 1. See the upstream-only changes after merge-base.
echo "=== upstream-only changes in ggml/include/ggml.h ==="
git diff --unified=3 "1fd6dfe9f3d4b69cce101d832339fbda2d14b056..upstream/master" -- "ggml/include/ggml.h" | sed -n '1,220p' || true

# 2. See the TurboQuant-only changes after merge-base.
echo "=== TurboQuant-only changes in ggml/include/ggml.h ==="
git diff --unified=3 "1fd6dfe9f3d4b69cce101d832339fbda2d14b056..HEAD" -- "ggml/include/ggml.h" | sed -n '1,220p' || true

# 3. Produce a three-way merge scaffold if conflict occurs after cherry-pick.
echo "=== three-way merge scaffold ==="
echo '# After a cherry-pick conflict on this file, run:'
echo '#   git checkout --ours  <file>   # or --theirs, depending on target'
echo '#   then apply only the TurboQuant-specific hunks below:'
