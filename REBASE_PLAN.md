# TurboQuant Rebase Plan — `feature/turboquant-kv-cache-rebase` (from `feature/turboquant-kv-cache`) → latest `upstream/master`

Source of truth: the exact refs recorded in the current parity audit below.
- Local HEAD: `71c3cc95d7a4` (`feature/turboquant-kv-cache-rebase`)
- Upstream/master: `876a4321163249c43ca4e986818fab5ab081f282`
- TurboQuant fork: `8a891f4b566efdbd3cea92fafee3227a0a267683` (`giveen/llama-cpp-turboquant`, `feature/turboquant-kv-cache`)
- Fork backup: `59145a4fa493695410438ea9a0071d0939c57619`
- merge-base: `1fd6dfe9f3d4b69cce101d832339fbda2d14b056`

All commit IDs below are verified exists in the local fork. **If upstream/master already contains the same function/module/test/docs, it is preferred and the TurboQuant patch should be skipped or reduced to the minimum delta that preserves TurboQuant semantics.**

---

# STATUS — 2026-07-31: THREE-WAY PARITY FIXES APPLIED; VALIDATION IN PROGRESS

> **This section supersedes the checklist below. Read this first.**

## Strict commit-level parity audit (2026-07-31, requirement = "ALL feature commits ported unless upstream already has same code")

- **Scope used for this audit:**
  - `HEAD` = `71c3cc95d7a4`
  - `upstream/master` = `876a43211632`
  - `giveen/feature/turboquant-kv-cache` = `8a891f4b566e`
- **Exact git results:**
  - `git rev-list --left-right --count HEAD...giveen/feature/turboquant-kv-cache` -> `605 366`
  - `git rev-list --left-right --count HEAD...upstream/master` -> `22 0`
  - `git cherry -v HEAD giveen/feature/turboquant-kv-cache | wc -l` -> `323`
- **Interpretation:**
  - Upstream is fully absorbed (`0` commits missing from upstream into HEAD).
  - The TurboQuant feature branch is **not** fully ported at strict commit/patch-equivalence level (`323` commits in fork branch not patch-equivalent in HEAD).
- **Current requirement status:** **NOT CONFIRMED** for "ALL commits ported unless upstream already does the same code".
- **Breakdown of those 323 non-equivalent feature commits (subject-prefix bucket):**
  - `fix`: 47
  - `feat`: 34
  - `perf`: 17
  - `experiment`: 27
  - `docs`: 25
  - `ci`: 6
  - `wip`: 2
  - `cleanup`: 1
  - `chore`: 1
  - `other/unprefixed`: 163

This means functional parity may be high in key runtime paths, but the stronger "all feature-branch commits are ported or upstream-equivalent" criterion is still open and needs a commit-by-commit resolution ledger.

## Complete commit audit artifact (2026-07-31)

Generated full ledger:
- `rebase-diff-recipes/turboquant-complete-audit.tsv`

Method used:
- Enumerated all commits in `1fd6dfe9f3d4..giveen/feature/turboquant-kv-cache`.
- For each commit, captured file list and tested reverse-patch applicability on HEAD (`git show <sha> | git apply --reverse --check -`) to detect whether the patch content is already present in the current tree.
- If reverse-apply failed, checked `git cherry -v HEAD giveen/feature/turboquant-kv-cache` for patch-equivalent commits already represented in local history.
- Added per-commit classification and code-touch flag.

Complete-audit counts (all 366 feature-branch commits):
- `PORTED_EXACT_TREE`: 77
- `PORTED_PATCH_EQ_HISTORY`: 92
- `UNRESOLVED_CODE`: 155
- `UNRESOLVED_NONCODE`: 42

Non-merge commit counts (326 commits; merge commits removed to reduce double counting):
- `PORTED_EXACT_TREE`: 77
- `PORTED_PATCH_EQ_HISTORY`: 73
- `UNRESOLVED_CODE`: 150
- `UNRESOLVED_NONCODE`: 26

Additional checks:
- For all `UNRESOLVED_CODE` commits, touched files still exist in the current tree (none were pure file-deletion false positives).
- `UNRESOLVED_CODE` hotspots are concentrated in Metal/CUDA/Vulkan and KV/graph paths (e.g. `ggml/src/ggml-metal/ggml-metal.metal`, `src/llama-kv-cache.cpp`, `ggml/src/ggml-cuda/fattn.cu`, `src/llama-graph.cpp`).
- Reverse-apply against an `upstream/master` worktree shows these unresolved commits are not trivially present in upstream tree either (all 197 unresolved are `not_in_upstream_tree` by reverse-apply).

Correctness checks executed for currently integrated TurboQuant paths:
- Built targets: `test-quantize-fns`, `test-backend-ops`, `test-turbo-quant`, `llama-bench`.
- Exit codes from the completed run: `test-quantize-fns=0`, `test-turbo-quant=0`, `test-backend-ops-cpu=0`.
- Log scans found no `FAIL` / `ERROR` / assertion/segfault markers in:
  - `/tmp/tq-audit/test-quantize-fns.log`
  - `/tmp/tq-audit/test-turbo-quant.log`
  - `/tmp/tq-audit/test-backend-ops-cpu.log`

Interpretation:
- The audit is complete in coverage (every feature-branch commit classified), but strict parity is still open due the unresolved-code bucket.
- To close strict parity, each `UNRESOLVED_CODE` row needs one explicit disposition: `PORTED_DIFFERENTLY`, `UPSTREAM_EQUIVALENT`, or `INTENTIONAL_DROP` with file-level evidence.

### Final closure pass (completed)

- The ledger now includes per-row `disposition` and `evidence` columns for every commit, including all `UNRESOLVED_CODE` rows.
- File: `rebase-diff-recipes/turboquant-complete-audit.tsv`
- `UNRESOLVED_CODE` rows annotated: `155 / 155` (no missing disposition/evidence fields).
- Final `UNRESOLVED_CODE` disposition counts:
  - `PORTED_DIFFERENTLY`: 115
  - `UPSTREAM_EQUIVALENT`: 0
  - `INTENTIONAL_DROP`: 40

Closure interpretation:
- Commit-by-commit annotation work is complete.
- Strict literal requirement (`all feature commits ported unless upstream-equivalent`) remains **NOT CONFIRMED** because 40 commits are explicitly marked `INTENTIONAL_DROP` and 0 are `UPSTREAM_EQUIVALENT`.
- Functional runtime parity may still be acceptable for the merge objective, but it is no longer represented as literal all-commit parity.

---

## Complete audit — code presence + correctness (2026-08-01, HEAD `71c3cc95d`)

> Supersedes the "strict parity remains open" note above: every `UNRESOLVED_CODE` commit has now been dispositioned with file-level evidence by a six-subsystem review (ggml-core/CPU, CUDA, Metal, Vulkan, llama-runtime, common/tools/tests). This section records the results and the resolution ledger for the 7 port regressions found.

### Method
- Per-commit presence: `git show <sha>` per unresolved commit, then grep/byte-diff of the introduced identifiers against the working tree.
- Per-file comparison fork tip (`8a891f4b5`) vs HEAD: 62 files byte-identical, 85 differ (deltas = upstream drift + the 7 regressions below), 20 fork-only dev artifacts (autoresearch, bench-smem, quality-gate scripts, investigation docs) intentionally absent.
- Turbo-token sweep (turbo/tq/wht/fwht/centroid/rotation identifiers) over all 167 files: only 3 files had missing tokens -> all three investigated and confirmed as regressions (findings 2-4).
- Runtime verification at HEAD before fixes: build OK; `test-turbo-quant` (turbo3 MSE=0/Cosine=1.0, turbo4 Cosine=0.9956) OK; `test-quantize-fns` OK; `test-backend-ops` CPU 17,825/17,825 OK, CUDA0 13,107/13,107 OK; end-to-end llama-cli decode on Qwen3-8B-Q8_0: f16 983/171, turbo2 865/162, turbo3 812/152, turbo4 794/157 t/s, coherent output, clean exit.

### Verdict summary
- All 155 `UNRESOLVED_CODE` commits accounted for: 153 present in HEAD (2 of them — `3eca09aec`, `a5a0f7b86` — are fork-internal reverts: the fork reverted them in `687b184f35`, so HEAD == fork tip is the correct end state), 0 absent, 2 superseded in-fork (q8_0 head-dim fallbacks `df33248d2`/`fb2d86d31` replaced by zero-padding `d158db5c6` before fork tip).
- Turbo codec (`ggml-turbo-quant.c`), CPU paths, CUDA kernels (10 files byte-identical), SYCL (5 commits ported), llama runtime wiring (38 commits, zero coding errors) all verified correct.

### The 7 port regressions (resolution ledger)

| # | Severity | Location | Finding | Resolution |
|---|----------|----------|---------|------------|
| 1 | BUG (crash, reproduced) | `ggml/src/ggml-cuda/ggml-cuda.cu:1911-1914` | TQ weight dispatch: `ggml_cuda_should_use_mmvq` (no TQ exclusion) runs before the fork's fused-TQ branch; TQ3_1S (and TQ4_1S with `GGML_TQ_NATIVE=1`) hits `GGML_ABORT` in `mmvq.cu:1148`. Fork's `!is_tq_weight` guard (`fork ggml-cuda.cu:2787`) dropped in the upstream rewrite of `ggml_cuda_mul_mat`. Default TQ4_1S masked by load-time q8_0 conversion. | **FIXED** — restored `!is_tq_weight` exclusion before the mmvq branch; verified `GGML_TQ_NATIVE=1` TQ4_1S mul_mat no longer aborts |
| 2 | CRITICAL (nil-pipeline crash) | `ggml/src/ggml-metal/ggml-metal.metal:11838/11926/11998` + `device.cpp:163-180` | `kernel_set_rows_turbo/turbo2/turbo4` defined but never `[[host_name]]`-instantiated; HEAD getter requests `kernel_set_rows_f32_i64_turbo3` (upstream naming) which exists nowhere -> NULL deref on every turbo KV write on Metal. Fork had 6 instantiations (`fork:12356-12371`). | **FIXED** — re-added 6 instantiations under HEAD naming (`kernel_set_rows_f32_i{32,64}_turbo{2,3,4}`) |
| 3 | CRITICAL (nil-pipeline crash) | `ggml/src/ggml-metal/ggml-metal.metal:11770-11800` | `kernel_get_rows_tq3_1s/tq4_1s` instantiations dropped (`fork:12307-12308`); GET_ROWS on TQ tensors on Metal -> NULL deref. | **FIXED** — re-added 2 instantiations |
| 4 | CRITICAL (abort) | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:5357-5367` | set_rows pipeline registration loop lacks TURBO2_0/3_0/4_0 (`fork:4952-4954`); shaders ARE generated (`vulkan-shaders-gen.cpp:848-850`) and bodies exist in `copy_to_quant.comp` (incl. wave64 ballot fix), but no `vk_pipeline` is created -> `GGML_ABORT` on every turbo KV write on Vulkan. | **FIXED** — re-added 3 registration lines with `require_full_subgroups=true, subgroup_size=32` |
| 5 | MEDIUM (feature loss) | `tools/server/server-context.cpp` | Fork's checkpoint sidecar (`eaf98e61`: PKCL magic, `.ckpt` save/load) and parallel-restore gating (`d6ae83f6` server half: `prompt_checkpoint_restored`) absent; upstream also lacks them; drop was undocumented. Cold-restart `action=restore` still discards restored state. | **FIXED** — ported both commits onto HEAD's refactored server (sidecar helpers + save/restore hooks; flag field/reset/batch-exclusion/restore-set/clear; the fork's per-token `break` adapted to HEAD's loop structure as a `prompt_checkpoint_restored` break inside the fill loop). Verified: server builds, save/restore endpoints execute cleanly (empty-slot round-trip + follow-up completion OK) |
| 6 | MEDIUM (test coverage) | `tests/test-backend-ops.cpp:8244,9900` | `all_types[]` lost TQ3_1S/TQ4_1S (`753f19982`); FA type_KV sweep lost TURBO3_0/TURBO4_0 (`88fcb67e5`/`bc5c9c891`, ~528 cases). TQ3_1S has zero backend-ops coverage; no FA test exercises turbo KV. | **FIXED** — restored both; suite now covers TQ3_1S/TQ4_1S in all_types sweeps and turbo3/4 FA (1056 FA cases on CUDA) |
| 7 | LOW | `common/chat-auto-parser-generator.cpp:168`, `tools/server/server-context.cpp:1302` | Auto-parser path does not tolerate leading whitespace before `<think>` (`8f3fbc0f` 4th hunk: `p.space() + p.optspace(start)`); `fb2d86d31`'s server-context "don't cap" hunk dropped (fallback semantics moved to zero-padding, server-side n_ctx cap hunk re-evaluated). | **FIXED** — auto-parser hunk ported verbatim; server no-cap behavior restored (warn + rely on rope scaling, matching fork) |

### Bonus fix (found while resolving #1): TQ4_1S native dp4a kernel broken

- **Symptom (reproduced):** with `GGML_TQ_NATIVE=1` (or any TQ3_1S model) the fixed dispatch reached the fused TQ kernel, which produced numerically garbage output (NMSE ~0.9-1.0) vs the CPU reference on every shape.
- **Root cause:** `tq4_cents8_reg()` in `ggml/src/ggml-cuda/mmvq-tq.cu` — the `__byte_perm`-based centroid LUT lookup. Empirically verified with standalone kernels on this toolchain (CUDA 13.3 / sm_120): `__byte_perm` with constant selectors folds with different semantics than runtime selectors (constant: 16-bit nibble selector; runtime: only selector bytes 0-1 honored -> result bytes 1,3 zero). The interleave constants (0x5140/0x7362) folded correctly, but the runtime-selector LUT steps (`__byte_perm(CR03, CR47, sel0)` with sel0 runtime) produced `0x81818181` for all inputs. The function is byte-identical to fork tip, i.e. **fork-inherited** (present since `51481c3b5`; the fork's native TQ4_1S decode was never numerically validated — its suite runs TQ4_1S as converted q8_0).
- **Fix:** rewrote the LUT with deterministic shifts (`cent()`: `((v & 8) ? (v & 4 ? CRCF : CR8B) : (v & 4 ? CR47 : CR03)) >> (8 * (v & 3)) & 0xFF`, sel0/sel1 interleaves via plain byte shifts), verified in isolation (all-nibble sweep correct), then in-suite.
- **Verified:** `GGML_TQ_NATIVE=1` TQ4_1S mul_mat (m=256, k=1536, n=1/2/4/8) now passes; scalar TQ3_1S/TQ4_1S paths unchanged (already correct).

### Resolution verification (2026-08-01, after all 8 fixes)

- Build: `llama-cli llama-server llama-quantize test-backend-ops llama-bench test-turbo-quant test-quantize-fns` — 0 errors, 0 warnings.
- `test-turbo-quant`: turbo3 MSE=0/Cosine=1.0, turbo4 Cosine=0.9956 — OK. `test-quantize-fns`: OK (incl. TQ3_1S/TQ4_1S).
- `test-backend-ops -b CPU`: **23,217/23,217 OK, 0 FAIL** (was 17,825 before the coverage fix; +5,392 restored cases incl. TQ3_1S/TQ4_1S all_types sweeps and the turbo3/4 FA sweep).
- `test-backend-ops -b CUDA0`: **16,311/16,311 OK, 0 FAIL** (was 13,107; +3,204). TQ3_1S MUL_MAT (fused scalar path) and TQ4_1S MUL_MAT (dp4a path, incl. `GGML_TQ_NATIVE=1`) all OK; 1,056 turbo FA cases run; SET_ROWS/GET_ROWS_BACK with TQ dst report not-supported (harness gate, fork-identical).
- End-to-end llama-cli on Qwen3-8B-Q8_0 (CUDA): f16 167, turbo2 157, turbo3 154, turbo4 156 t/s generation, coherent output, clean exit.
- llama-server with `--slot-save-path`: save/restore endpoints execute cleanly (round-trip + follow-up completion OK); sidecar code paths exercised (no checkpoints persisted in the empty-slot flow — upstream prompt-cache migration moves prompts between slots; the fork's checkpoint-populated scenario needs the fork's mid-conversation flow).

### Non-regressions (verified, not fixed)
- `SET_ROWS_TURBO3/4` "not supported [CUDA0]" in the harness: fork-identical (supports_op byte-identical); the graph path dispatches turbo set_rows via `ggml_cuda_op_set_rows` and end-to-end decode works.
- Fork-inherited latent issues, intentionally left byte-identical to fork tip: 64KB stack array `float G[TURBO_D*TURBO_D]` (`ggml-turbo-quant.c:81`, `a5a0f7b86` reverted in fork), dead 64-group quantize path (`blocks_per_group = 0` for group_size=64, unreachable due to 128-padding), `TURBO4_USE_4BIT=0` rnorm read-after-no-write (dead since 4-bit default), stale comments.
- CUDA `get_alloc_size` divergence (upstream's in-buffer f16_extra reservation kept while launch_fattn pool-allocates): over-reservation only, no functional impact; left as-is to minimize delta (fork's removal was performance-motivated).
- 42 `UNRESOLVED_NONCODE` commits: docs/README/CI/merge commits; CI workflows ported under `present_exact_or_equivalent_in_tree` commits; README prebuild links are fork branding, intentionally dropped.
- 14 fork-only files: dev artifacts (autoresearch, bench-smem, turbo-quality-gate, ROCm notes) intentionally dropped.

## What was done (this session)

- **The rebase/merge is committed, with follow-up parity fixes currently in the working tree.** The union of the TurboQuant fork and `upstream/master` is on branch `feature/turboquant-kv-cache-rebase`.
  - Current HEAD before these follow-up edits: `edcbe53c5` (`docs : record full parity sweep results (fork + upstream vs tree)`)
  - upstream base in tree: `876a4321` (`vulkan: add POOL_1D op (#25431)`)
  - current fork tip audited: `8a891f4b5` (`vulkan: fix turbo3 signs ballot packing on wave64 subgroups (#241)` plus merge commit)
  - Working tree contains the three parity fixes in `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-quants.c`, and `tools/server/server-context.cpp`, plus this document.
  - The previously-uncommitted 170-file tree was committed in 7 logical commits on top of `ed572867c`: `53f2170da` (ggml core), `421f6a45c` (cuda), `58c733800` (sycl), `cb1cc8566` (vulkan/metal/hip), `d37584432` (llama/common/models), `2128110c2` (docs + rebase recipes), `7f05310b5` (ci workflows). `533e25c84` (`WIP: add TurboQuant KV cache types`, cherry-pick of fork `0b3744874`) and `ed572867c` predate them and are the branch's first two commits.
- **RE-REBASED onto latest upstream (2026-07-31):** `git fetch upstream` showed the base `5f55650a7` was 16 commits behind the remote tip `876a4321`. All 9 TurboQuant commits were replayed onto `876a4321` (`git rebase --onto upstream/master 5f55650a7`). **Only one conflict**, in `src/llama-context.cpp` (the `d37584432` port commit), caused by upstream `69e62fc77` "llama : enforce the same K and V cache types for DeepSeek V4; enable FA if V cache is quantized (#25871)". Resolution: kept upstream's MLA/DEEPSEEK4 K==V enforcement and its quantized-V FA auto-enable/error; moved the TurboQuant turbo-type FA auto-enable BEFORE upstream's quantized-V check so turbo V + explicit `--flash-attn off` still auto-enables (fork semantics preserved); dropped the fork's now-dead duplicate quantized-V error. Pre-rebase tip preserved at `backup/rebase-tip-7f05310b5`. All other 8 commits replayed cleanly (incl. the SYCL and Vulkan commits overlapping upstream's new SYCL/vulkan work).
- **The full build succeeds, 0 errors:**
  - `llama-cli` ✅ `llama-server` ✅ `llama-quantize` ✅ `test-backend-ops` ✅ `llama-bench` ✅
  - Command: `cmake --build build --target llama-cli llama-server llama-quantize test-backend-ops llama-bench -j $(nproc)`
  - Log: `/tmp/build-port16.log` (all `[100%] Built target ...`)
  - Binaries present in `build/bin/` (incl. `libggml-cuda.so`, `libllama.so.0.0.10202`)
- **Smoke tests pass:**
  - `llama-server --version` → `10225 (f987a49f8)` ✅ (re-verified 2026-07-31 after the re-rebase; earlier builds were `10202 (ed572867c)` / `10209 (7f05310b5)` — same tree content, new base)
  - `llama-bench` detects the CUDA device (`RTX 5090, compute capability 12.0`) ✅
  - `llama-cli --help`, `llama-quantize --help`, `llama-bench --help` all run ✅
  - `test-backend-ops -b CPU` runs ops (e.g. IM2COL_3D ... OK) ✅
- **All 5 TurboQuant types are registered in `ggml/include/ggml.h`:**
  - `GGML_TYPE_TURBO2_0 = 43`, `GGML_TYPE_TURBO3_0 = 44`, `GGML_TYPE_TQ3_1S = 45`, `GGML_TYPE_TQ4_1S = 46`, `GGML_TYPE_TURBO4_0 = 47`
- **18 merge-artifact defects were found and fixed** by the iterative build-fix loop: 16 stacked-duplicate / dropped-symbol defects plus 2 bonus restorations (the `auto_flid` init and the `turbo-rotation-data.h`/`-32.h` headers). Full work log in [§7 Work Log](#7-work-log--the-18-merge-artifact-fixes-16-primary--2-bonus) below.
- **DSPARK support RESTORED (2026-07-31)** — user decision (reversed the earlier strip). DSPARK is **upstream-merged code**: commit `84075273c` "spec: add DSpark speculative decoding (#25173)" (2026-07-28) is an ancestor of `upstream/master` (verified via `git merge-base --is-ancestor` against ref tip `5f55650a`, 2026-07-30 — see Decision 4); the earlier "uncommitted PR" belief was disproven. The fork itself has zero DSPARK, so DSPARK exists in this tree only as upstream code. All DSPARK code was restored from upstream across all 13 affected files. Verified: every file now matches upstream's `dspark` reference count exactly; the 5-target build passes; DFlash semantics preserved. **Important distinction:** `DGX-Spark` (NVIDIA GB10 / compute-capability 12.10 hardware support: `GGML_CUDA_CC_DGX_SPARK`, UMA handling) is a SEPARATE, upstream-merged feature and was never touched. See §7 Scope Decisions + Decision 4.
- **No leftover conflict markers** in `src/`, `common/`, `ggml/`, `tools/` (the `=======` hits are ASCII art in comments/logs, not merge markers).
- **Parity fixes applied after the prior audit:** restored `GGML_OP_TURBO_WHT` meta-backend split handling, restored TurboQuant row-validation cases, and restored server handling for DFlash/Eagle3 convenience flags during output sizing and speculative initialization.
- **Current parity status:** upstream is fully contained in the local tree; strict all-commits parity with the TurboQuant feature branch is not yet confirmed and requires commit-by-commit disposition evidence.

## What still needs to be done

- [ ] **RUNTIME validation with a real model** — **PARTIALLY DONE (2026-07-31 evening):** turbo2/3/4 KV verified on Qwen3-8B-Q8_0 + Qwen3.5-9B-UD (bench + end-to-end llama-cli decode); DFlash speculative decoding verified (Qwen3.6-35B-A3B-DFlash draft + KAT-Coder-2.5-Dev-Q5_K_S main, turbo3 V, small ctx). **dsv4 VERIFIED (2026-07-31):** DeepSeek-V4-Flash-0731-UD-Q3_K_XL (230 GB, partial offload via `--fit on -ngl auto`) loads and generates on this build with both q8_0/q8_0 and turbo3/turbo3 KV (MLA requires same K/V type per upstream 69e62fc77), 27-chunk stream + `[DONE]` at ~12 t/s. **BLOCKED — DSPARK draft run:** the only DSpark drafts on disk (`ds4flash-dspark.gguf`, `ds4flash-dspark-q4k.gguf`) use a custom arch `deepseek4-dspark` with `mtp.N.*` latent-attention tensor names and nested `mtp.2.markov_head.markov_w1/w2` + `confidence_head.proj` tensors. NO llama.cpp ref (fork, upstream, or this tree) supports that arch/format — upstream's DSpark (the restored #25173 code) expects Qwen3-style `dflash`-arch drafts with top-level `markov_w1`/`markov_w2`/`conf_proj` tensors, and no such draft model exists here. The Qwen3.6-35B-A3B-DFlash draft was confirmed to have NO markov tensors (pure DFlash). Making the dsv4 drafts load would be new feature work (port a `deepseek4-dspark` arch), not a rebase fix; the DSPARK code path itself is upstream's and compiles/passes token parity.
  - Bench results (Qwen3-8B-Q8_0, CUDA, after the cudart fix — see Bugs section): f16 pp=3574/tg=152, turbo2 pp=3546/tg=150, turbo3 pp=12110(!)/tg=157, turbo4 pp=3461/tg=147. Qwen3.5-9B: turbo3 pp=4144/tg=117, turbo4 pp=4172/tg=117.
  - DFlash run: main 35B-A3B Q5_K_S (23G) + draft 402M, `--spec-type draft-dflash -ctv turbo3 -c 1024`, generated correctly, clean exit.
- [x] **`llama-quantize` CLI type table + `llama-bench` parser — DONE (2026-07-31):** added `TQ3_1S`/`TQ4_1S` rows to `tools/quantize/quantize.cpp` (fork-exact rows, `--help` now lists them) and `tq3_1s`/`tq4_1s` to `llama-bench.cpp::ggml_type_from_name()`. Group 1 GAP closed.
- [x] **DSPARK: RESTORED (2026-07-31)** — `deepseek4-dspark.cpp` exists in **no ref** in this repo (fork included); the earlier "MISSING" claim was incorrect. DSPARK is **upstream-merged code**: commit `84075273c` "spec: add DSpark speculative decoding (#25173)" (2026-07-28) is an ancestor of `upstream/master`. The fork itself has zero DSPARK, so the code exists here only as upstream code and is kept in this tree. All DSPARK code was **restored** from upstream across 13 files (see §7 Scope Decisions + Decision 4). Every file now matches upstream's `dspark` ref counts. `DGX-Spark` hardware support was never touched.
- [x] **`docs/KV-cache-quantization.md` — ADDED (2026-07-31):** written fresh (neither fork nor upstream ever had it); documents turbo2/3/4 usage, TQ3_1S/TQ4_1S weight types, rotation/padding rules, env knobs. `docs/speculative.md` ✅ exists.
- [x] **`tests/test-turbo-quant.c` — RE-ADDED (2026-07-31):** restored from fork tip (67 lines, incl. the #59 inverse-WHT round-trip fix) and registered via `llama_build_and_test(test-turbo-quant.c LABEL "turbo")` in `tests/CMakeLists.txt`. Passes: turbo3 basis-vector MSE=0/Cosine=1.0, turbo4 Cosine=0.9956.
- [ ] **Commit the follow-up parity fixes** — code and documentation are currently uncommitted; review and validation are required first.
- [x] **`common/console.cpp` termios gating — VERIFIED survived (2026-07-31)** — POSIX init gated on `if (!simple_io)` at `common/console.cpp:135-151`, cleanup gated identically at `:159-166`; `tcgetattr`/`tcsetattr` only touch the terminal on the interactive path.
- [ ] **Metal / Vulkan / SYCL runtime validation** — the ported code compiles for CUDA/CPU; Metal (`ggml-metal.metal` TurboFlash), Vulkan (turbo3/4 FA + SET_ROWS), and SYCL template-instance files are present (now tracked/committed) but were **not runtime-validated** on this Linux/CUDA host.
- [x] **Full `test-backend-ops` suite — PASSES (2026-07-31):** `-b CUDA0 -j 8` = 12,718 OK / 0 FAIL / 7,663 not-supported / clean exit, after fixing the `ggml_get_n_tasks` merge artifact (see Bugs item 5). Note: `-b CUDA` (without the device index) silently skips everything — the filter matches the device name `CUDA0` exactly. The CUDA0 run exercises the CPU backend as reference for every case, covering both backends.
- [ ] **Baseline `f16` load check** — done implicitly (f16 bench + f16 cli runs pass); explicit §4 protocol still pending.

## Verification audit (2026-07-31, post-commit) — plan vs committed tree

Re-verified the full plan against the committed tree (HEAD `7f05310b5`, tree clean). Everything checked below is evidence, not re-derivation; the only doc-vs-code mismatches found are the ones this document now corrects (noted inline below and in the groups they affect).

**Verified matching (spot list):**
- Types: `GGML_TYPE_TURBO2_0..TURBO4_0` = 43/44/47, `TQ3_1S/TQ4_1S` = 45/46 in `ggml/include/ggml.h`; type traits registered in `ggml/src/ggml.c:767-801+`; CPU vec_dot + traits in `ggml/src/ggml-cpu/ggml-cpu.c`; codec in `ggml/src/ggml-turbo-quant.c` (byte-identical to fork tip `59145a4fa`). `ggml-quants.c` contains NO turbo code (see §6 correction).
- Ops: `GGML_OP_TURBO_WHT` in `ggml.h:578` + `ggml.c` name/compute; `ggml/include/ggml-rpc.h` `static_assert(GGML_OP_COUNT == 102)` with the TURBO_WHT/LIGHTNING_INDEXER/DSV4_HC_* comment.
- CUDA: `convert.cu` dequant dispatch for all 5 types, `fattn-vec`/`fattn-mma-turbo`/`fattn-tile` instances in `CMakeLists.txt`, `mmvq-tq.cu`, `turbo-innerq.cu`, `turbo-wht.cu`, `set-rows.cu` — present. SYCL: `turbo-wht.cpp/.hpp`, `set_rows.cpp`, `fattn-vec.hpp`, `turbo-quant.hpp` + 24 instance files. Metal: `turbo-matrices.h`, `turbo-wht.h`, TurboFlash p1/p2 kernels, `TURBO_SPARSE_V` skip branch, `mul_mm_tq_rotated`. Vulkan: `dequant_tq4_1s.comp`, `dequant_turbo3_0.comp`, `mul_mat_vec_tq4_1s.comp`, `turbo_wht.comp` + FA dequant shaders.
- Runtime wiring: `get_k_idx` both classes (`llama-kv-cache.cpp:1688/3181`), `auto_flid` (`llama-cparams.h:42`, init `llama-context.cpp:250`), `turbo-rotation-data.h`/`-32.h` restored, `TURBO_LAYER_ADAPTIVE` (`llama-kv-cache.cpp:317-341`), `TURBO_AUTO_ASYMMETRIC` gate (`:153`), inverse-WHT post-processing in `llama-graph.cpp` (FA + non-FA paths), single clean `llm_graph_input_dsv4*` copies in `llama-graph.h`, `selected_experts_in` restored, `common/speculative.cpp` upstream 3-arg-ctor impl + `static_assert(COUNT == 11)` (`:2411`).
- DSPARK: `dflash.cpp` dspark ref count 20/20 vs upstream `5f55650a7` (diff = one removed blank line); `build_dspark_markov_head` (`:125`) + tensor block (`:46-54`); `llama-model.h:620-624` fields; `llama-arch.h/.cpp` tensors; `common/common.h` enum + `need_n_rs_seq` (`:392`); `speculative.cpp` map entry, `is_dspark`, n_max clamp, type_to_str, DSPARK factory case (`:2471-2474`); `gguf-py/gguf/constants.py` + `tensor_mapping.py`; `conversion/qwen.py:692-694` `Qwen3DSparkModel`; `conversion/__init__.py:56`; `docs/speculative.md` DSpark section + `--spec-type` list + table; `tools/server/README.md`, `tools/cli/README.md`.
- `src/models/eagle3.cpp` byte-identical to fork tip (Decision 2 holds). `kv_cache_types` in `common/arg.cpp:311-314` (turbo2/3/4). ftype round-trip: `llama.h:159-160` -> `llama-quant.cpp:834-835` -> `llama-model-loader.cpp:752-753`. CI: `tqp-release.yml` + `tqp-linux-release.yml`. No conflict markers in `src/`, `common/`, `ggml/`, `tools/`.
- Token-level fork-vs-HEAD sweep: 37 key files, regex `turbo|tq|wht|dspark|iswa|flid` identifiers from fork tip ALL present in HEAD — 0 missing tokens (no dropped TurboQuant code).
- Build: `cmake --build build --target llama-cli llama-server llama-quantize test-backend-ops llama-bench` succeeds at HEAD (0 errors, binaries relinked); `llama-server --version` = `10209 (7f05310b5)`.

**Doc corrections applied this session:** branch name + HEAD + committed-tree status (above), Group 1 GAP refined (plumbing complete, CLI table only), Group 2 `f82f4083d` nix = N/A (HEAD `nix/` is upstream's, no duplicate spirv-headers entry), Group 10 rotation env-knob names corrected, Group 11 ui commits marked NOT merged, §5 current-state line, §6 File Map rows (`ggml-quants.c`, `tools/quantize/`, `tools/ui/dist`, `console.cpp`), §8 risks (uncommitted-work risk resolved).

**Post-rebase re-verification (historical, 2026-07-31 evening):** a token sweep over the same 37 files found 0 missing fork tokens; the 5-target build passed on the then-current base; the conflict resolution in `src/llama-context.cpp` preserved both upstream behavior and the fork's turbo auto-enable. The current verification is recorded in the status and parity sections above.

## Bugs found & fixed during validation (2026-07-31) — uncommitted, awaiting review

1. **llama-cli EOF spin (upstream bug at base `876a4321`, fixed in `tools/cli/cli-context.cpp`).** New client-style cli (conversation mode) spins at ~1M `"> "` writes/sec on stdin EOF: the interactive loop does `if (buffer.empty()) continue;` with no EOF check. Reproduced on f16 AND turbo KV (not turbo-related); gdb showed `console::readline_advanced` + `cli_context::run()`. Fix: `if (buffer.empty()) { if (std::cin.eof() || feof(stdin)) { break; } continue; }`. Now `llama-cli ... </dev/null` generates and exits cleanly. (Both readline paths return false on EOF; the empty-buffer signal was being discarded.)
2. **gguf-py/gguf/constants.py stacked-duplicate merge artifacts (fixed).** `import gguf` hard-crashed (duplicate `MODEL_TENSOR` members: `HC_HEAD_FN/BASE/SCALE`, `FFN_GATE_TID2EID`, `ATTN_COMPRESSOR_*`, plus a 10-member fork block stacked on upstream's identical members; duplicate `Keys` entries; shadowing `Keys.HyperConnection` class; duplicate `MODEL_ARCH_NAMES`/`TENSOR_NAMES`/`MODEL_TENSOR_SKIP` DEEPSEEK4 entries — the fork's generic DEEPSEEK4 skip was shadowing upstream's detailed one). All deduplicated keep-first/keep-upstream; fork's genuine additions (TQ types, DSPARK tensors, HC tensors) intact; `import gguf` now works (443 MODEL_TENSOR members). Same artifact class as §7 items 3-4 which only fixed `src/llama-arch.h`.
3. **CUDA first-launch failures — root cause: build linked system cudart 12.4 with CUDA 13.3-compiled code (fixed at build level).** Symptoms: `CUDA error: invalid argument` from SOFT_MAX / mul_mat_q mm_ids_helper on the FIRST launch of a kernel after `CUDA_SET_SHARED_MEMORY_LIMIT`; DFlash runs crashed 100%, non-draft runs survived only because a later launch cleared the sticky error. Root cause: `cudaGetDeviceProperties` via cudart 12.4 + driver 595 (CUDA 13.2) returns garbage for `sharedMemPerBlockOptin`/`sharedMemPerMultiprocessor` (4294967297 vs correct 101376 via `cudaDeviceGetAttribute`) → the max-dynamic-shared attribute set to garbage poisons the next launch (invalidValue once). Fix: `cmake -B build -DCUDA_CUDART=/usr/local/cuda-13.3/targets/x86_64-linux/lib/libcudart.so -DCUDA_cudart_LIBRARY=...` (the stale `CUDA_CUDART` cache var pointed at `/usr/lib/x86_64-linux-gnu/libcudart.so` = 12.4; `linkLibs.rsp` carries the absolute path). After relink: `DT_NEEDED libcudart.so.13`. **Side effect: pp512 on Qwen3-8B jumped 3545 -> 12110 t/s** — the stale runtime was silently crippling CUDA (graph/launch fallbacks). **Hazard for fresh builds on this machine:** any clean reconfigure reverts to system cudart 12.4 unless the cache var is set; also `libcublas.so.12` (12.4) remains in use and is fine (talks to the driver directly).
4. **`Qwen3.6-35B-A3B-UD-Q4_K_S-rot-kv.gguf` does not load (NOT a bug).** `expected 813 tensors, got 733` — the file carries 80 baked-in per-layer tensors `blk.N.attn_k_rot.weight` / `attn_v_rot.weight` (40 layers x 2) produced by an external rotation-experiment conversion pipeline. NO llama.cpp ref (fork or upstream) has tensor mappings for them. Used the compatible `Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_S.gguf` (same qwen35moe arch/vocab) as the DFlash main model instead.
5. **`ggml_get_n_tasks` merge artifact — NULL deref for single-input ops (fixed, `ggml/src/ggml-cpu/ggml-cpu.c`).** The union merge fused two case groups into one and applied the RWKV/GATED_LINEAR_ATTN body (`n_heads = node->src[1]->ne[1]`) to the whole group — including single-input ops (ARGSORT, TOP_K, PAD, ROLL, ARANGE) whose `src[1]` is NULL. `test-backend-ops -b CUDA0` segfaulted in `ggml_graph_plan` at the first such test (caught deterministically; gdb backtrace). Fixed by splitting per upstream/fork: the UPSCALE/PAD/ROLL/ARANGE/ARGSORT/TOP_K/FLASH_ATTN*/SSM_* group gets `n_tasks = n_threads`; RWKV/GATED keeps the head-count body. **Full suite now: 12,718 OK / 0 FAIL / clean exit (was crash at ~8,293 OK).**

### Validation results (2026-07-31, all on RTX 5090 / CUDA)
- Turbo KV: bench f16/turbo2/3/4 on Qwen3-8B (pp 3574/3546/12110/3461, tg 152/150/157/147 t/s), turbo3/4 on Qwen3.5-9B-UD (pp 4144/4172, tg 117). End-to-end llama-cli decode on turbo2/3/4 (after fix #1).
- DFlash: `--spec-type draft-dflash` with Qwen3.6-35B-A3B-DFlash-Q8_0 draft + 35B-A3B Q5_K_S main, `-ctv turbo3 -c 1024 -n 24` — generated a correct answer, clean exit. (Crash before fix #3.)
- test-turbo-quant: passes (see above). test-backend-ops CUDA SET_ROWS/SOFT_MAX: no matching cases in upstream matrix ("Skipping" by design).

## Full parity sweep (2026-08-01) — TheTom fork + upstream vs this tree

**Upstream (ggml-org/llama.cpp):** local `upstream/master` == remote tip `876a4321163249c43ca4e986818fab5ab081f282`; it is fully contained in this branch. The local op enum is upstream plus `GGML_OP_TURBO_WHT`; the local type enum contains the five TurboQuant types with upstream-compatible symbolic mappings. The current tree also includes the fork-equivalent meta-backend WHT split case.

**Fork (`giveen/llama-cpp-turboquant`):** the exact `feature/turboquant-kv-cache` tip is `8a891f4b566efdbd3cea92fafee3227a0a267683`, two commits past backup `59145a4fa`:
- `11a8377bd` vulkan turbo3 wave64 ballot fix (#241) — **PORTED (2026-08-01)** to `copy_to_quant.comp` (was missing: our tree had the buggy `.x`-shift packing).
- `0b059740a` HIP FA pool bypass — already in tree (plan Group 3 `0757ff4ee`).

**File inventory:** 84 fork files absent from HEAD — ALL verified zero TurboQuant tokens: upstream-removed/refactored files (hexagon htp, fattn-wmma, clamp/sin/cos/sqrt/square shaders, json/regex-partial, unary_gelu, export-graph-ops, get-model, ui sidebar), fork-dev artifacts (autoresearch scripts, bench-smem, quality-gate, TURBOQUANT_UPSTREAM_MERGE.md, rocm test notes, dsv4-flash jinja), and fork's `ggml-cpu/dsv4-ops.cpp` (content merged into `ggml-cpu/ops.cpp`).

**Two-way token sweep (all 2,988 common files):** forward gaps (fork tokens missing in HEAD) found ONLY in fork test files + README/scripts — **tests now ported** (see below); README/scripts are fork-branding/dev artifacts intentionally not ported. Reverse gaps (HEAD tokens absent in fork) = DSPARK only, expected (upstream-merged, fork has none).

**Tests ported (2026-08-01):** `test-backend-ops.cpp` — TURBO_WHT fwd/inv + round-trip, SET_ROWS turbo3/turbo4/TQ4_1S round-trips, TQ4_1S mul_mat sweeps, TQ4_1S tolerance case (suite: 12,936 OK / 0 FAIL; turbo3/4 SET_ROWS reported unsupported on CUDA — identical to fork, those tests target Vulkan). `test-quantize-fns.cpp` — rotated-domain buffer sizing, turbo KV skip with rationale, TQ3_1S thresholds.

**Verdict (updated for strict requirement):** upstream tip containment is confirmed, and major TurboQuant runtime surface is present; however, strict "ALL feature commits ported unless upstream-equivalent" is **not yet confirmed**. The 2026-07-31 strict audit found 323 feature-branch commits not patch-equivalent in HEAD, so this plan must keep an open commit-resolution ledger until each one is classified as either "ported", "upstream-equivalent", or "intentionally dropped" with evidence.

## Decisions recorded during the merge (deliberate "reduce to minimum delta" choices)

1. **`common/speculative.cpp`: kept upstream's `common_speculative_impl_draft_dflash`** (3-arg ctor with `type` param) and **dropped the fork's older copy** (with `StashedG`/`m_use_deferred` deferred-KV-injection fields). The merged `common/common.h` and the impl factory are upstream-driven. Whole-tree grep confirmed **zero** external references to the dropped `StashedG`/`m_use_deferred`/`MAX_STASH`. *(2026-07-31: the DSPARK factory case that uses the 3-arg ctor was restored — see §7 Scope Decisions; the `type` ctor param defaults to DFLASH except in the restored DSPARK factory case, kept for upstream parity.)*
2. **`src/models/eagle3.cpp`: restored the fork's version wholesale** — the union merge had produced a broken hybrid (upstream's `target_layer_ids`/`n_embd_tgt` formula spliced onto fork's `LLM_KV_EAGLE3_*` keys). All 16 symbols the fork's file references were verified present in the merged tree. Note: upstream's `norm_before_fc` support in eagle3 was consequently dropped.
3. **`auto_flid` was silently disabled in the merged tree** until this session added both the `bool auto_flid;` field in `src/llama-cparams.h` and the `cparams.auto_flid = true;` default init in `src/llama-context.cpp` (matches upstream exactly). Without this, fused Lightning Indexer would never auto-resolve.
4. **DSPARK: RESTORED from upstream (2026-07-31).** DSPARK is **upstream-merged code**: commit `84075273c` "spec: add DSpark speculative decoding (#25173)" (2026-07-28) is an ancestor of `upstream/master`, verified via `git merge-base --is-ancestor` against the local ref tip `5f55650a` (2026-07-30); the earlier "uncommitted PR" belief was disproven. The fork itself has zero DSPARK, and `deepseek4-dspark.cpp` exists in **no ref** in this repo; DSPARK exists in this tree only as upstream code. All DSPARK code was restored across 13 files (Group 9); every file now matches upstream's `dspark` ref counts, and the 5-target build passes. **Distinguish from DGX-Spark:** `GGML_CUDA_CC_DGX_SPARK` / UMA handling in `ggml-cuda` is NVIDIA hardware support (compute capability 12.10 / GB10), upstream-merged, and was never touched. **Hedge:** no `dgx-spark` *model* code exists in any ref of this repo — only the NVIDIA hardware macro — so if "dgx-spark" refers to a draft model, it is neither present nor ported.
5. **`src/llama-context.cpp` re-rebase conflict (2026-07-31 evening):** upstream `69e62fc77` (enforce same K/V cache types for MLA/DEEPSEEK4; auto-enable FA for quantized V, error on explicit DISABLED) collided with the fork's turbo FA auto-enable block in `llama_new_context_with_model`. Kept upstream's checks; moved the fork's turbo-type auto-enable before them (turbo V + explicit DISABLED still auto-enables, fork semantics); dropped the fork's duplicate quantized-V error (upstream's version is stricter and earlier).

---

## 1. Subsystem Grouping

### Group 1 — Quantizer Registration + Convertor
**STATUS: ✅ COMPLETED (code merged; Group 1 GAP closed 2026-07-31 — CLI table rows + bench parser entries added)**

The commits that add new `ggml_type` enum values and `llama-quantize` support.

- [x] `5bad823b1` `Update GGMLQuantizationType and LlamaFileType enums to include TQ3_1S and TQ4_1S quantization types with corresponding sizes in GGML_QUANT_SIZES.` — verified in `ggml/include/ggml.h` (TQ3_1S=45, TQ4_1S=46)
- [x] `25a19f223` `feat: GGML_TYPE_TURBO2_0 — 2-bit TurboQuant KV cache (6.4x compression)` — `GGML_TYPE_TURBO2_0 = 43` present
- [x] `6c9cfb1be` `experiment: split 2x4-entry constant LUT for M1 decode fix`
- [x] `5e6277b6f` `fix: add turbo3/turbo4 cache types to llama-bench arg parser` — `turbo2/turbo3/turbo4` present in `tools/llama-bench/llama-bench.cpp:503-509`
- [x] `f3f7c3c4b` `feat: InnerQ per-channel equalization + turbo2 64-group fallback`
- [x] **GAP CLOSED (2026-07-31):** added `tq3_1s`/`tq4_1s` to `llama-bench.cpp::ggml_type_from_name()` and `TQ3_1S`/`TQ4_1S` rows to the `tools/quantize/quantize.cpp` CLI table (fork-exact rows, `llama-quantize --help` now lists them).

Rebuild verification:
```bash
build/bin/llama-quantize --help | grep -E 'TQ3_1S|TQ4_1S'
build/bin/test-quantize-fns   # TQ3_1S/TQ4_1S coverage passes
```

---

### Group 2 — GGML Core + WHT Ops
**STATUS: ✅ COMPLETED (code merged; `ggml/src/ggml-turbo-quant.c`, WHT op, turbo-quant headers present; SYCL `turbo-wht.cpp` tracked/committed)**

Must exist before runtime wiring or CUDA dispatch.

- [x] `012faec26` `feat: add GGML_OP_TURBO_WHT — custom O(d log d) Walsh-Hadamard Transform`
- [x] `377727552` `feat: replace dense 128x128 matvec with Fast Walsh-Hadamard rotation #26`
- [x] `7173be941` `perf: optimized turbo3 dequant — eliminates context scaling regression`
- [x] `a1fafdc44` `fix(meta): add GGML_OP_TURBO_WHT to tensor-split path (#196)` — restored in the current tree with the fork-equivalent meta-backend case; GGML_OP_COUNT assert updated (ggml-rpc.h handled in this session)
- [x] `f82f4083d` `fix(nix): remove duplicate spirv-headers entry (#195)` — **N/A in final tree (2026-07-31):** HEAD `nix/` is upstream's untouched; no duplicate `spirv-headers` entry exists there. Fork's nix changes were not ported (upstream nix kept).
- [x] `bf590c723` `fix(turbo-quant): add forward declaration for turbo_cpu_fwht_inverse`

Pitfall: if `tools/llama-bench/llama-bench.cpp:ggml_type_from_name()` in upstream already maps these enums, Group 1 is the only place you need parser patches.

---

### Group 3 — CUDA/HIP Kernel Dispatch
**STATUS: ✅ COMPLETED (build passes with `libggml-cuda.so`; CUDA layer was the first part of the build to pass)**

Backend op registration for turbo KV types. Apply only if upstream lacks dispatch.

- [x] `70de24909` `fix(cuda): allow f16/bf16 + q8_0 mixed KV without GGML_CUDA_FA_ALL_QUANTS (#82)`
- [x] `58bbe5518` `fix(cuda): add F16-K + TURBO-V dispatch cases in fattn.cu`
- [x] `fa4e8be0a` `fix(cuda): add F16-K + TURBO-V dispatch cases in fattn.cu` *(alias)*
- [x] `fec0719a3` `vulkan: add SET_ROWS support for turbo2_0 and turbo4_0 (#50)`
- [x] `ed81ed03e` `CUDA/HIP: implement get_rows for TQ4_1S and TQ3_1S`
- [x] `e69af784a` `fix(fattn): add (turbo*, F16) template instantiations`
- [x] `5aeb2fdbe` `fix(hip): add (turbo*, F16) template-instance .cu files to HIP build`
- [x] `7e341660d` `fix(perplexity): cast n_ctx * nv to size_t in KL logits save (#138)`
- [x] `3c0efbdc6` `HIP/MUSA: fix build break from unguarded 3D peer memcpy and bare cudaEventCreate (#173)`
- [x] `0757ff4ee` `fix(hip): bypass pool for FA f16 temp buffers to prevent OOM`
- [x] `4d754604e` `fix: force VEC FA path for quantized KV on HIP/ROCm`
- [x] `d7b533446` `fix(hip): bypass pool for FA f16 temp buffers to prevent OOM` *(alias)*
- [x] `8993d4fd7` `fix: force VEC FA path for quantized KV on HIP/ROCm` *(alias pair)*
- [x] `f2dc968bd` `cuda: disable sparse V skip (warp divergence regression)`
- [x] `11a241d0d` `Merge pull request #105 from TheTom/fix/disable-sparse-v-cuda`

Verification:
```bash
./build/bin/test-backend-ops -p 'SET_ROWS_TURBO*' >/tmp/set-rows.log   # empty output is expected
```

---

### Group 4 — Runtime Wiring: `llama-context`, `llama-kv-cache`, `llama-graph`
**STATUS: ✅ COMPLETED (build passes). Session fixes in this area:**
- `src/llama-context.cpp` — removed duplicate `llm_fused_op_probe` block; added missing `cparams.auto_flid = true;` init
- `src/llama-kv-cache.cpp` — re-inserted dropped `llama_kv_cache_context::get_k_idx` definition (link-stage catch); `turbo-rotation-data.h` / `turbo-rotation-data-32.h` restored from fork
- `src/llama-graph.cpp` — removed stacked duplicate dsv4 section; removed duplicate `llm_graph_result::add_fused_node`
- `src/llama-graph.h` — removed stacked duplicate classes; restored `using llm_graph_cb` typedef; restored `selected_experts_in` param in `build_moe_ffn` decl2

Function-level targets:

**`src/llama-context.cpp`**
- [x] `k_is_turbo` / `v_is_turbo` head_dim zero-padding guards
- [x] flash-attn auto-enable block for turbo types

**`src/llama-kv-cache.cpp`**
- [x] GQA auto-asymmetric gate
- [x] layer-adaptive type checks / boundary V rewrites
- [x] rotation matrix creation; K/V head_dim padding
- [x] `get_k` / `get_v` padded-head return path
- [x] `cpy_k` / `cpy_v` zero-padding + WHT group `op_params` writeback

**`src/llama-graph.cpp`**
- [x] flash-attn and non-FA inverse WHT post-processing path

**direct commits touching these files from the verified list:**
- [x] `db3595a75` `fix(kv-cache): per-side env-knob control for upstream attn rotation, default OFF`
- [x] `b6f8e7f72` `fix(llama-graph): n_head_v reshape uses Q-head count, not KV-head count (#78)`
- [x] `c452be605` `fix: disable upstream attn rotation by default (conflicts with TurboQuant)`
- [x] `2f756e67e` `fork: reconcile MTP lineage with TurboQuant+ KV cache`
- [x] `e8f23eb76` `docs: real Metal benchmarks after #include fix — 8× gap not 35× #23`
- [x] `d044965e5` `feat: MSE-only mode — drop QJL, all 3 bits to PolarQuant #23`

Verification:
```bash
./build/bin/llama-bench -m /path/to/model.gguf -ngl 99 --cache-type-k q8_0 --cache-type-v turbo3 -c 4096 -b 1 --n-gen 32 --no-host 1   # ⚠️ NOT YET RUN (no model available)
```

---

### Group 5 — Flash-Attn / FA Kernel Instantiations
**STATUS: ✅ COMPLETED (build passes with CUDA; template instances for turbo2/3/4 × f16/q8_0 present under `ggml/src/ggml-cuda/template-instances/`, `ggml/src/ggml-sycl/template-instances/`)**

TurboQuant-specific FA kernels for CUDA/HIP/Metal/Vulkan/SYCL. **⚠️ HIGH CONFLICT RISK** because upstream `74976e1ae` and `90e0f5cfc` reshaped CUDA dispatch and fused-ops layout.

- [x] `157f27f71` `perf: turbo VEC flash attention — +9% decode on CUDA via autoresearch`
- [x] `a17a63a38` `fix: VEC flash-attn Q/K stride mismatch in vec_dot_fattn_vec_KQ_turbo3_0`
- [x] `e9ab0452b` `fix: graceful fallback for turbo3 with non-128-aligned head dims (issue #13)`
- [x] `fb2d86d31` `fix: graceful fallback for turbo3 on non-128-aligned head dims (issue #13)` *(alias)*
- [x] `107362298` `fix: add TURBO2_0 to flash_attn auto-enable check`
- [x] `93bf21d7a` `fix(metal): add turbo2/3/4 types to FLASH_ATTN_EXT and CPY support checks`
- [x] `93bf21d7a` / `0198d5819` `vulkan: fix turbo3 build + coopmat FA after April upstream sync`
- [x] `458c7f102` `vulkan: fix turbo3 build + coopmat FA after April upstream sync` *(alias)*
- [x] `539ce5de9` `fix(fattn): gate turbo2 fused-MMA decode to head_dim 128`
- [x] `4e223ee9a` `fattn: extend fused MMA decode to turbo3 + turbo2 (parity on turbo4)`
- [x] `38dc66596` `turbo4 MMA decode: default ON (GQA-packed tensor-core path)`
- [x] `b3e51cf3d` `feat(fattn): fused turbo4 MMA decode path (opt-in)`
- [x] `77ab7e988` `turbo4: correct Lloyd-Max centroids, drop dead rnorm 68->66B`
- [x] `545092c36` `turbo3 centroids: re-derive to exact Lloyd-Max optimum (match reference)`
- [x] `bf9bf3180` `fix: TQ4_1S CUDA — mmvq exclusion for fused path + quantize tool registration` *(TQ weight variant)*
- [x] `76ebd2650` `fix: TQ4_1S on MoE models — disable CUDA graphs for TQ MUL_MAT_ID`
- [x] `6479b0b7f` `SYCL: add TurboQuant KV cache support for Intel GPUs`

---

### Group 6 — Vulkan + Vulkan-Turbo-FA
**STATUS: ⚠️ MERGED (shader files tracked/committed: `dequant_tq4_1s.comp`, `dequant_turbo3_0.comp`, `mul_mat_vec_tq4_1s.comp`, `turbo_wht.comp`), NOT runtime-validated on this host**

Apply only if Vulkan backend is in scope.

- [x] `a494833d0` `feat: Vulkan compute shader support for turbo3 KV cache`
- [x] `ff8bb7394` `vulkan: fix and complete turbo3 KV cache support`
- [x] `88fcb67e5` `vulkan: add turbo3 backend tests`
- [x] `035ac80d55` `tom/vulkan-turbo-fa` cleanup context trail
- [x] `175a652dd` `vulkan: sync turbo4 centroid tables with the re-derived C reference`
- [x] `03ff84818` `vulkan: drop removed rnorm field in turbo4_0 block (fix KV layout mismatch)`
- [x] `bc5c9c891` `tests: cover turbo4 in SET_ROWS round-trip and flash-attn cross-backend cases`
- [x] `a09bafedd` `vulkan: restore turbo_wht op + turbo3/4 FA dispatch (regression from b9190 upstream sync)`
- [x] `36dc8e2ec` `rpc : update GGML_OP_COUNT assert for TURBO_WHT op (fixes RPC builds)`
- [x] `7d5ac40c2` `vulkan: fix botched-merge FA shader preprocessor + gate turbo FA`
- [x] `f94c3b840` `vulkan: force f32 accumulation for quantized K/V flash attention`
- [x] `fd1be2101` `vulkan: submit more frequently on integrated GPUs to avoid device-lost`
- [x] `2f8661751` `vulkan: fuse TurboQuant K/V dequant into flash attention`

---

### Group 7 — Metal / Apple Silicon
**STATUS: ⚠️ MERGED (code present: TurboFlash kernels in `ggml-metal.metal`, `mul_mm_tq_rotated` pipeline, FLASH_ATTN_EXT/CPY type checks), NOT build/runtime-validated (no macOS host)**

Apply only if Metal backend is in scope.

- [x] `a494833d0` `feat: Vulkan compute shader support for turbo3 KV cache` *(not Metal; listed for cross-ref)*
- [x] `e596f476f` `metal: add TurboFlash attention kernel for turbo3 KV cache decode`
- [x] `a4736ffbd` `Add GitHub Sponsors funding link`
- [x] `93bf21d7a` `fix(metal): add turbo2/3/4 types to FLASH_ATTN_EXT and CPY support checks`
- [x] `67f076f2e` `fix(metal): disable TurboFlash by default — corrupt output on Apple10`
- [x] `a1bcb34a1` `fix(metal): disable TurboFlash by default — corrupt output on Apple10` *(merge)*
- [x] `d3271ac41` `fix: gate turbo V unpad on V type + disable TurboFlash on Apple10 (#91)`
- [x] `b8b1d49b3` `fix(hip/metal): add missing f16-turbo fattn-vec instances`
- [x] `b01afefed` `metal: drop write to removed rnorm field in turbo4_0 (fixes Apple Silicon startup crash)`
- [x] `c452be605` `fix: disable upstream attn rotation by default (conflicts with TurboQuant)`
- [x] `7f23abad6` `fix(metal): set ne12/ne13/r2/r3 function constants in mul_mm_tq_rotated pipeline`

---

### Group 8 — TQ3_1S / TQ4_1S Weight Quantization Layer
**STATUS: ✅ COMPLETED (code merged; CLI type-table entries and focused quantization tests pass).**

These are first-class model-weight compression types in this fork. **Required, not optional.**

- [x] `3a2fad19c` `feat: TQ3_1S + TQ4_1S weight quantization with V2.1 fused Metal kernels`
- [x] `2ebd51963` `feat: CUDA port of TQ4_1S/TQ3_1S weight dequant (signalnine)`
- [x] `8c2e0d878` `perf: fused TQ4_1S/TQ3_1S mul_mat_vec — 3.4x decode speedup`
- [x] `adda3bc84` `fix: TQ4_1S on MoE models — disable CUDA graphs for TQ MUL_MAT_ID`
- [x] `76ebd2650` `perf: V12 single-phase fused TQ mmvq — shmem activation, no global scratch`
- [x] `51481c3b5` `perf: TQ4_1S native kernel 3.5× faster — 240 t/s (was 68), smaller VRAM`
- [x] `cc1bae215` `perf: warp-cooperative TQ4_1S dequant (16× less compute per block)`
- [x] `941d4567f` `feat: multi-token TQ4_1S dp4a kernel + multi-GPU fix + static build fix`
- [x] `579db2981` `fix: replace __dp4a with ggml_cuda_dp4a for HIP/ROCm compatibility`
- [x] `0bf1eef8d` `fix: AMD/RDNA4 arch dispatch — scalar half path for TQ4_1S on AMD GPUs`
- [x] `c29fab6e8` `Enhance Metal operations for TQ weights and concurrency handling for Gemma 4`
- [x] `753f19982` `feat: add MoE expert count kernel instantiations + TQ4_1S backend tests`
- [x] `657160439` `fix: cap map0 kernel shmem for 256-expert MoE models`
- [x] `71c7a4ced` `fix: remove redundant extern from GGML_API macro (GCC 13.3 hard error)`
- [x] `fe2ead962` `Fix turbo4 C reference WHT dequant mismatch (#43)`
- [x] `e3ce079be` `feat: load-time TQ4_1S -> q8_0 conversion for CUDA dp4a speed`
- [x] `296259217` `fix: add dk512 Metal FA kernel instances for turbo types (Gemma 4 support)`
- [x] `b1a6f79b3` `fix: CPU vec_dot heap allocation for turbo/TQ types (n > 4096 models)`
- [x] `694ed0314` `fix: AMD HIP/ROCm build support for TQ4_1S weight compression`
- [x] `06a6b6226` `experiment: batched byte extraction + explicit bit field pre-extract`

Backends covered: Metal, CUDA, HIP/ROCm. Template instance coverage is intentionally not enumerated line-by-line; re-create from `ggml/src/ggml-cuda/template-instances/` and `ggml/src/ggml-sycl/template-instances/`.

Verification:
```bash
build/bin/llama-quantize --help | grep -E 'TQ3_1S|TQ4_1S'
build/bin/test-quantize-fns   # includes TQ3_1S/TQ4_1S coverage
./build/bin/llama-bench -m <tq-weighted-model.gguf> -ngl 99 -c 4096 -b 1 --n-gen 32 --no-host 1
```

---

### Group 9 — DeepSeek-V4-Flash / DSPARK Speculative Draft Port
**STATUS: ✅ COMPLETED — upstream DFlash + DSPARK speculative draft ported (`src/models/dflash.cpp` present with DSpark Markov head); DSPARK RESTORED from upstream (2026-07-31)**

**Upstream has DFlash (`d1b34251b`) including the DSpark Markov-head variant (`84075273c` #25173). The fork itself has zero DSPARK. User decision (2026-07-31, then reversed the same day): DSPARK is RESTORED and kept — all of its code was brought back from upstream. `DGX-Spark` hardware support (a separate, upstream-merged feature) is retained.**

- [x] `5d0c2929c` `spec : add DFlash support (#22105)` — upstream has this; merged (dflash.cpp present)
- [x] `a4fd3adbd` `dflash: refactor draft model conversion (#25110)`
- [x] `9c1ddef4d` `llama : add guard for K/V rotation input when buffer is unallocated (#25215)`
- [x] `212b49cfb` `spec: support spec-draft-p-min in DFlash (#25246)`

**DSPARK restore completed (13 files, all match upstream `dspark` ref counts):**
- [x] `src/models/dflash.cpp` — DSpark tensor-loading block, `build_dspark_markov_head()`, and build-graph call restored
- [x] `src/llama-arch.h/.cpp` — `LLM_TENSOR_DSPARK_MARKOV_W1/W2/CONF_PROJ` restored (enum + name map + layer ops)
- [x] `src/llama-model.h` — `dspark_markov_w1/w2`, `dspark_conf_proj/_b` fields restored
- [x] `common/common.h` — `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` enum entry + `need_n_rs_seq` ref restored
- [x] `common/speculative.cpp` — `draft-dspark` map entry, `is_dspark` member/ctor/sampling-branch, type_to_str + n_max switch cases, factory case; `static_assert(COUNT == 11)`
- [x] `gguf-py/gguf/constants.py` — `DSPARK_MARKOV_W1/W2` + `DSPARK_CONF_PROJ` enum, name map, DFLASH arch list entries restored
- [x] `gguf-py/gguf/tensor_mapping.py` — DSPARK mappings restored
- [x] `conversion/qwen.py` / `conversion/__init__.py` — `DSparkModel` / `Qwen3DSparkModel` restored
- [x] `docs/speculative.md`, `tools/cli/README.md`, `tools/server/README.md` — `draft-dspark` docs restored
**Note:** DSPARK is upstream-merged code (commit `84075273c` #25173, dated 2026-07-28, is an ancestor of `upstream/master`). DSPARK is kept in this tree. See Decision 4. **Runtime consequence (reversed):** a DSpark-format GGUF checkpoint (DFLASH arch + `markov_w1.weight`/`conf_proj` tensors) is **supported again** — those tensors are registered and such checkpoints load. **Validation finding (2026-07-31):** the DSPARK code path could not be exercised end-to-end — the only DSpark drafts on disk (`ds4flash-dspark.gguf`, `ds4flash-dspark-q4k.gguf`) use a custom `deepseek4-dspark` arch with `mtp.N.*` latent-attention tensors + nested `mtp.2.markov_head.markov_w1/w2`/`confidence_head.proj`, which no llama.cpp ref supports; upstream's DSpark expects Qwen3-style `dflash` drafts with top-level `markov_w1`/`markov_w2`/`conf_proj`, and the Qwen3.6-35B-A3B-DFlash draft in /mnt/storage/models has no markov tensors. **`DGX-Spark` hardware support (NVIDIA GB10 / CC 12.10) is a separate upstream-merged feature and was never touched.**

---

### Group 10 — Runtime Feature Flags: Layer-Adaptive, Boundary V, Sparse V, TurboFlash, Rotation Override
**STATUS: ✅ MERGED (code present per Group 4 session fixes), ⚠️ runtime behavior not yet validated**

These are behavioral toggles and runtime paths that must survive rebase even though they are scattered across runtime code rather than isolated commits.

| Feature | Runtime location | Resolution rule | Status |
|---------|------------------|-----------------|--------|
| Layer-adaptive KV (`TURBO_LAYER_ADAPTIVE` env + mode 5/6/7) | `src/llama-kv-cache.cpp` | Preserve mode enum + auto-enable logic | ✅ merged |
| Boundary V | `src/llama-kv-cache.cpp` boundary-first/last layer assignments | Append to upstream `llama_kv_cache` constructor | ✅ merged |
| Sparse V skip (`TURBO_SPARSE_V=0` opt-out) | `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-metal/ggml-metal.metal` | Re-add skip branch if removed | ✅ merged |
| TurboFlash / Metal FA | `ggml/src/ggml-metal/*` | Keep additive | ✅ merged (unvalidated) |
| Upstream rotation override (env: `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE`, hard lock-out `LLAMA_ATTN_ROT_DISABLE`) | `src/llama-kv-cache.cpp` rotation env-knob path (`:551-607`) | Default OFF (TurboQuant manages rotation) | ✅ merged, verified |

---

### Group 11 — Server / Bench CLI / UI / Docs / CI Hygiene
**STATUS: ⚠️ PARTIAL — server/bench parity fixes are merged in the working tree; fork UI commits remain intentionally unmerged because upstream is preferred; docs and TurboQuant tests are present.**

Apply last; lowest conflict risk.

- [x] `514c6c768` `server: gate speculative init/warnings without dropping seq_rm probe`
- [ ] `f27268914` `ui: constrain chat inset height so long messages can scroll` — **NOT MERGED (2026-07-31):** HEAD `tools/ui/` is byte-identical to upstream `5f55650a7`; the fork's ui tweaks were dropped. Skipped per the upstream-preference rule; revisit only if the chat-inset behavior matters for the TurboQuant+ release.
- [ ] `c09b7c879` `ui: don't trust a partial dist dir; validate full asset set before embedding` — **NOT MERGED** (see above). Note: `tools/ui/dist/` is gitignored in both fork and HEAD; `embed.cpp` falls back to an empty asset table without it.
- [ ] `3a45046f9` `ui: validate complete asset set in HF download before proceeding` — **NOT MERGED** (see above).
- [x] `4503343ff` `docs: point prebuilt table at tqp-v0.3.0`
- [x] `bb81d3334` `docs: add TurboQuant+ prebuild links`
- [x] `558c6b78e` `docs: add Linux prebuild links`
- [x] `99c2ed759` `ci: add TurboQuant+ release workflow`
- [x] `4d24ad87b` `ci: add TurboQuant+ release workflow` *(v0.1.1)*
- [x] `8ba9f1288` `vulkan: TQ4_1s support for model weights (#69)`
- [x] `07595065d` `fix: inverse WHT in test-turbo-quant.c round-trip (#59)` — `tests/test-turbo-quant.c` restored and passing in the current tree
- [x] `f03d33144` `vulkan: TQ4_1s support for model weights (#69)` *(v0.1.0)*

---

## 2. Verified Cherry-Pick Recipe

> **NOTE (2026-07-31):** the rebase was performed as a union merge of the fork tree onto upstream, then committed as a linear series: `533e25c84` (WIP cherry-pick of fork `0b3744874`) -> `ed572867c` -> six port commits -> docs -> ci. This recipe is retained as historical reference and for replaying any missing pieces (e.g. the Group 1 TQ CLI table rows, the docs/test-turbo-quant gaps, the Group 11 ui commits).

```bash
set -euo pipefail
REPO=/mnt/storage/rebase-tq
cd "$REPO"

# 0. Remotes
git remote add upstream https://github.com/ggerganov/llama.cpp.git 2>/dev/null || true
git fetch upstream --tags --prune

# 1. Fresh base
git checkout -B feature/turboquant-kv-cache upstream/master

# 2. Replay in dependency order; prefer upstream when the same
#    function/module/test/docs already exists upstream.

#    2a. quantizer + convertor registration
for h in 5bad823b1 25a19f223 6c9cfb1be 5e6277b6f f3f7c3c4b; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

#    2b. GGML WHT ops
for h in 012faec26 377727552 7173be941 a1fafdc44 f82f4083d bf590c723; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

#    2c. CUDA/HIP kernel dispatch
for h in 70de24909 58bbe5518 fa4e8be0a fec0719a3 ed81ed03e e69af784a 5aeb2fdbe 7e341660d 3c0efbdc6 0757ff4ee 4d754604e f2dc968bd 11a241d0d; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

#    2d. Runtime wiring + FA corrections (function-targeted ahead of cherry-pick)
for h in db3595a75 b6f8e7f72 c452be605; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

#    2e. FA / kernel quality/correctness fixes
for h in 157f27f71 a17a63a38 107362298 93bf21d7a 0198d5819 458c7f102 \
         539ce5de9 4e223ee9a 38dc66596 b3e51cf3d 77ab7e988 545092c36; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

#    2f. Vulkan + turbo-FA (optional by platform)
# for h in a494833d0 ff8bb7394 88fcb67e5 175a652dd 03ff84818 bc5c9c891 \
#          a09bafedd 36dc8e2ec 7d5ac40c2 f94c3b840 fd1be2101 2f8661751; do
#   git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
# done

#    2g. Metal (optional by platform)
# for h in e596f476f 93bf21d7a 67f076f2e d3271ac41 b8b1d49b3 b01afefed 7f23abad6; do
#   git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
# done

#    2h. SYCL (optional by platform)
# for h in 6479b0b7f d604b432c ae6e8577f 2fa44d526 236699777; do
#   git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
# done

#    2i. TQ weight quant layer — REQUIRED, not optional
for h in 3a2fad19c 2ebd51963 8c2e0d878 adda3bc84 76ebd2650 \
         51481c3b5 cc1bae215 941d4567f 579db2981 0bf1eef8d \
         c29fab6e8 753f19982 657160439 71c7a4ced fe2ead962 \
         e3ce079be 296259217 b1a6f79b3 694ed0314 06a6b6226; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

#    2j. dflash speculative draft (DSPARK restored from upstream — kept in tree)
for h in a4fd3adbd 9c1ddef4d 212b49cfb; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done
# DSPARK: user decision 2026-07-31 (reversed same day) — RESTORED. DSPARK is
# upstream-merged code (commit 84075273c #25173 is an ancestor of upstream/master);
# the code is present here only as upstream code and was restored across 13 files
# (see REBASE_PLAN §7 Scope Decisions). DGX-Spark hardware support is a separate
# feature, retained.

#    2k. runtime feature flags — function-targeted patches
#        layer-adaptive, boundary V, sparse V, TurboFlash, rotation override
# See Group 10 table in REBASE_PLAN.md for exact files and rules.

#    2l. server / ui / docs / ci
for h in 514c6c768 f27268914 c09b7c879 3a45046f9 4503343ff bb81d3334 \
         558c6b78e 99c2ed759 4d24ad87b 8ba9f1288 07595065d f03d33144; do
  git cherry-pick -x "$h" || { echo "ABORT at $h"; exit 1; }
done

# 3. Rebuild after each group
cmake --build build --target llama-cli llama-server llama-quantize test-backend-ops llama-bench -j "$(nproc)"
nm -C build/bin/llama-quantize | grep -E 'turbo[234]|TQ[34]_1S' || true
```

---

## 3. Conflict Resolution Recipes

**This is the primary lever for moving success probability from ~70% to ~90%.**
When a cherry-pick conflicts in the high-risk files below, do not blindly rerun. Use the preflight recipe scripts in `/mnt/storage/rebase-tq/rebase-diff-recipes/` to see exact upstream vs TurboQuant deltas, then apply only the TurboQuant-specific function targets.

Run a recipe:
```bash
bash /mnt/storage/rebase-tq/rebase-diff-recipes/src_llama-kv-cache_cpp.sh | less
```

### Highest-risk conflict files + resolution rule
| File | Resolution rule |
|------|-----------------|
| `src/llama-graph.cpp` | Accept upstream's `ggml_mul_mat_aux` refactor; re-add TurboQuant's inverse-WHT post-processing as a new helper called after upstream's path, not by overwriting `ggml_mul_mat_aux`. |
| `src/llama-kv-cache.cpp` | Upstream added `llama_kv_cell` metadata changes; preserve upstream cell layout and patch only the turbo-specific `get_k`/`get_v`/`cpy_k`/`cpy_v` padded-head returns and WHT `op_params`. |
| `src/llama-context.cpp` | Upstream added `llm_fused_op_probe` table; append TurboQuant turbo types to the existing probe array rather than replacing it. |
| `ggml/include/ggml.h` | Merge enum blocks: keep upstream order, append TQ3_1S/TQ4_1S immediately after the highest existing turbo enum. Do not reorder existing entries. |
| `ggml/src/ggml-quants.c` | Upstream added Q2_0 quantization tables; append turbo/TQ tables after the existing quantizer arrays. Do not duplicate existing entries. |
| `src/models/dflash.cpp` | Upstream has DFlash (`d1b34251b`). Use upstream's draft-model impl as the base and apply only TurboQuant-specific `bonus_anchor` guards on top. The DSpark Markov-head code is kept in the tree (restored from upstream, 2026-07-31). |
| `tools/llama-bench/llama-bench.cpp` | Upstream reshaped args handling. Patch only the `ggml_type_from_name()` return mapping; leave upstream argparse structures untouched. |
| `common/console.cpp` | Apply termios gating as an additive guard only; do not touch upstream console init/display logic. |
| `common/speculative.cpp` | Upstream added DFlash + speculative plumbing; append TurboQuant draft-type cases rather than replacing enum blocks. |

### When to stop and ask
If a conflict affects the function signature, struct layout, or call graph of upstream shared code (not just TurboQuant-specific blocks), stop cherry-picking that group and switch to function-targeted patches from the recipe. Do not force a cherry-pick through with `-X theirs` on shared files.

---

## 4. Validation Artifact Rules

- `f16` baseline must load cleanly before any turbo comparison. **⚠️ not yet executed (no model file).**
- Capture exact `t/s`, VRAM, startup time per build.
- `stat build/bin/llama-bench`, `build/bin/llama-cli`, `build/bin/llama-server`, `build/bin/libggml-cuda.so.0`, `build/bin/libllama.so.*`.
- `test-backend-ops -p 'SET_ROWS_TURBO1'` matches 0 tests by design. Not proof of runtime success.

### Failure taxonomy before rerunning
- `tcsetattr: Inappropriate ioctl for device` → `common/console.cpp` termios gating
- `cannot run the operation (SET_ROWS)` → missing CUDA dispatch for the type
- `GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor)) failed` → allocator padding patch
- Generic `failed to create context` without assert text → instrument `llama_init_from_model` for `err.what()` + backtrace

---

## 5. Merge Base

- Commit: `1fd6dfe9f3d4b69cce101d832339fbda2d14b056`
- Confidence: medium (squash-merges obscure true ancestor)
- Confirm post-construction with: `git merge-base HEAD upstream/master`
- Current state: branch `feature/turboquant-kv-cache-rebase`, base commit `edcbe53c5`, upstream `876a4321`, fork tip `8a891f4b5`; the working tree contains the uncommitted parity fixes and this document update pending review.

---

## 6. File Map: All Files That Must Be Patched

| File | Reason | Status |
|------|--------|--------|
| `ggml/include/ggml.h` | `ggml_type` enum entries for turbo quantizers + TQ3_1S/TQ4_1S | ✅ present (types 43-47) |
| `ggml/src/ggml-quants.c` | **Correction (2026-07-31): no turbo code here.** Quantizer tables/type traits live in `ggml/src/ggml.c` (`:767-801` etc.), CPU vec_dot + traits in `ggml/src/ggml-cpu/ggml-cpu.c`, codec entry points in `ggml/src/ggml-turbo-quant.c`, declarations in `ggml/src/ggml-quants.h:109-119+`. All present. | ✅ present (in ggml.c / ggml-cpu.c / ggml-turbo-quant.c) |
| `ggml/src/ggml-turbo-quant.c` | WHT, codec entry points | ✅ present |
| `ggml/src/ggml-cuda/concat.cu` | Contiguity/quantized concat for turbo tensors | ✅ present |
| `ggml/src/ggml-cuda/set-rows.cu` | SET_ROWS turbo dispatch | ✅ present |
| `ggml/src/ggml-cuda/convert.cu` | get_rows for TQ3_1S/TQ4_1S | ✅ present |
| `ggml/src/ggml-cuda/fattn.cu` | F16-K + TURBO-V dispatch cases | ✅ present |
| `ggml/src/ggml-cuda/fattn-vec.cu` | turbo* template instantiations | ✅ present |
| `ggml/src/ggml-cuda/mvvq.cu` / `mmvq.cu` | TQ weight fused mmvq | ✅ present |
| `ggml/src/ggml-metal.metal` | Metal turbo decode kernels | ✅ present (unvalidated) |
| `ggml/src/ggml-vulkan.c` / shaders | Vulkan SET_ROWS + turbo3/4 FA | ✅ present (unvalidated) |
| `ggml/src/ggml-sycl.mm` | SYCL TurboQuant KV support | ✅ present (unvalidated) |
| `src/llama-context.cpp` | Turbo guards, head_dim padding, flash-attn auto-enable | ✅ fixed this session |
| `src/llama-kv-cache.cpp` | Asymmetric gate, layer-adaptive types, rotated V boundary, WHT op_params | ✅ fixed this session (get_k_idx restored) |
| `src/llama-graph.cpp` | Inverse-WHT post-processing | ✅ fixed this session |
| `src/models/dflash.cpp` + `models.h` | DFlash draft path | ✅ present |
| `src/models/deepseek4-dspark.cpp` | DSPARK draft path | ✅ **RESTORED** (never existed in any ref; kept as upstream code in `dflash.cpp`, 2026-07-31) |
| `common/speculative.cpp` | Draft-type enum plumbing | ✅ fixed this session |
| `tools/quantize/quantize.cpp` | Convert/quant entry + calibration gate | ✅ TQ3_1S/TQ4_1S table rows added 2026-07-31 (`--help` lists them; `--type 43/44` numeric fallback also works) |
| `tools/llama-bench/llama-bench.cpp` | `ggml_type_from_name()` token→enum | ✅ `tq3_1s`/`tq4_1s` added 2026-07-31 (had turbo2/3/4) |
| `gguf-py/gguf/*.py` | DFlash tensor mappings + bonus-anchor keys | ✅ DSPARK mappings restored |
| `conversion/qwen.py` / `conversion/__init__.py` | DSpark/DSPARK mappings | ✅ `DSparkModel`/`Qwen3DSparkModel` restored |
| `common/console.cpp` | termios gating for non-TTY scripted runs | ✅ verified 2026-07-31 — POSIX block gated on `!simple_io` (`:135-151`, cleanup `:159-166`) |
| `docs/KV-cache-quantization.md`, `docs/speculative.md` | User-facing instructions | ✅ KV-cache-quantization.md ADDED 2026-07-31 (written fresh — existed in neither fork nor upstream); speculative.md ✅ |
| `tools/ui/dist/*` | Prebuilt static assets | ⚠️ **not in tree (2026-07-31)** — gitignored in both fork and HEAD (`tools/ui/.gitignore:12`); produced at release-build time by `embed.cpp` (empty asset table fallback). Fork's ui source tweaks (Group 11) were NOT ported; HEAD `tools/ui/` == upstream. |
| `.github/workflows/*` | TurboQuant+ release workflow | ✅ present |
| `tests/test-turbo-quant.c` | WHT round-trip test | ✅ RE-ADDED 2026-07-31 from fork tip + registered (`llama_build_and_test` LABEL "turbo"); passes |

---

## 7. Work Log — The 18 Merge-Artifact Fixes (16 primary + 2 bonus)

> The union merge stacked duplicate blocks and dropped a few definitions. Each was found
> via the iterative build-fix loop (`build-port*.log`), verified byte-identical against
> `upstream/master` and/or `backup/fork-tip-59145a4fa` where relevant, then fixed
> deterministically. All 16 are COMPLETED and the build passes.

1. ✅ **`src/llama-graph.cpp`** — removed stacked duplicate dsv4 section (kept fork's copy, which uniquely contains `llm_graph_input_attn_kv_iswa`; both copies verified byte-identical to upstream AND fork).
2. ✅ **`src/llama-hparams.h`** — removed duplicate dsv4 field block (8 fields incl. `dsv4_compress_ratios`).
3. ✅ **`src/llama-arch.h`** — removed 26 duplicate `LLM_KV_`/`LLM_TENSOR_` enum entries (keep-first dedup; enum is name-keyed so safe); inserted missing `LLM_ARCH_MINIMAX_M3` and `LLM_ARCH_NANBEIGE` at upstream's exact positions.
4. ✅ **`src/llama-graph.h`** — removed stacked duplicate classes (`llm_graph_input_dsv4_raw`, `llm_graph_input_dsv4`, `llm_graph_fused_node`); removed duplicate `llm_graph_result` members (`add_fused_node`/`get_fused_nodes`/`fused_nodes`); restored the `using llm_graph_cb` typedef that the dedup accidentally consumed; removed duplicate `build_inp_dsv4` declaration.
5. ✅ **`src/llama-kv-cache.h`** — removed duplicate `get_layer_ids`/`get_k_storage` declarations (fork's copy).
6. ✅ **`src/llama-arch.cpp`** — removed duplicate `case LLM_ARCH_DEEPSEEK4:`.
7. ✅ **`src/llama-model.h`** — removed duplicate `wo_a` and the entire stacked second DeepSeek-V4 per-layer tensor block.
8. ✅ **`src/llama-context.cpp`** — removed duplicate `llm_fused_op_probe` struct + 7 probe constants (byte-identical copies).
9. ✅ **`src/models/models.h`** — removed duplicate `struct llama_model_deepseek4` (byte-identical, 114 lines).
10. ✅ **`src/llama-cparams.h`** — added missing `bool auto_flid;` field (upstream has it; merged `llama-context.cpp` references it).
11. ✅ **`src/llama-graph.h`** — restored missing `selected_experts_in` parameter in `build_moe_ffn` decl2 (fork's header had dropped it; the merged .cpp definition + call site use upstream's 24-arg signature).
12. ✅ **`src/llama-model.cpp`** — inserted missing `} break;` closing the `GLM_DSA` case in `create_memory` (the merge had lost it, nesting the `DEEPSEEK4` case inside `GLM_DSA` — the root cause of the whole-file brace imbalance).
13. ✅ **`src/llama-graph.cpp`** — removed duplicate `llm_graph_result::add_fused_node` definition (byte-identical; kept upstream-positioned copy).
14. ✅ **`src/models/eagle3.cpp`** — restored fork's version wholesale (the merge produced a broken upstream/fork hybrid referencing undeclared `target_layer_ids`/`n_embd_tgt`; all 16 fork-symbol dependencies verified present in merged tree).
15. ✅ **`common/speculative.cpp`** — removed fork's duplicate `common_speculative_impl_draft_dflash` (kept upstream's 3-arg-ctor version; whole-tree grep confirmed the dropped `StashedG`/`m_use_deferred` had zero external refs). *(2026-07-31: the DSPARK references in this struct were restored — see §7 Scope Decisions.)*

**Scope decisions (not merge-artifact defects):**

- ✅ **DSPARK restore (2026-07-31)** — user decision (reversed the earlier strip same day). Restored all DSPARK code across 13 files (see Group 9): DSpark Markov head in `dflash.cpp`, `LLM_TENSOR_DSPARK_*` in `llama-arch.h/.cpp`, `dspark_*` fields in `llama-model.h`, `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` + `is_dspark` logic in `common/common.h` + `common/speculative.cpp` (enum 10 -> 11, incl. the factory case that was false-skipped by the restore script's idempotency marker and hand-fixed), gguf-py tensors/mappings, `conversion/qwen.py` `DSparkModel`, and `draft-dspark` docs (incl. `--spec-type` list + table row). Verified: every file matches upstream's `dspark` ref counts exactly (dflash.cpp 20/20, speculative.cpp 16/16, docs 9/9, etc.); 5-target build passes; Python syntax OK; DFlash semantics preserved (`n_draft_max = block_size-1`, `n_block_tokens = n_draft+1`, greedy branch kept). DSPARK is upstream-merged code (commit `84075273c` #25173 is an ancestor of `upstream/master`). `DGX-Spark` hardware support was never touched.
16. ✅ **`src/llama-kv-cache.cpp`** — re-inserted dropped `llama_kv_cache_context::get_k_idx` definition (declared in header, defined in upstream at :2847-2848, absent in merged .cpp — surfaced at link stage).

**Bonus fixes:**
- ✅ `src/llama-context.cpp` — added `cparams.auto_flid = true;` default init (was silently missing; fused Lightning Indexer would never auto-resolve).
- ✅ Restored `src/turbo-rotation-data.h` / `src/turbo-rotation-data-32.h` from the fork (referenced by `llama-kv-cache.cpp` but missing from the merge).

---

## 8. Risks

- **Validation is the bottleneck, not authorship.** Expect backend op failures from the Turboquant failure taxonomy.
- **Toolchain hazard (2026-07-31):** this machine's CUDA toolchain is mixed — compile uses CUDA 13.3 (`/usr/local/cuda-13.3`), but a stale `CUDA_CUDART`/`CUDA_cudart_LIBRARY` cache var links system cudart 12.4 (`/usr/lib/x86_64-linux-gnu`), which returns garbage device props with driver 595 and caused first-launch CUDA failures (fixed via cache var override — see Bugs section; a fresh `cmake -B build` must re-apply `-DCUDA_CUDART=...` or the DFlash/first-launch crashes return).
- **Model-format caveat (2026-07-31):** `*-rot-kv.gguf` files carry baked-in `attn_k_rot`/`attn_v_rot` tensors that no llama.cpp ref (fork or upstream) maps — they do not load (`expected 813, got 733`). Not a code bug.
- **Cross-port drift.** Do not port `src/models` by whole-file copy; use backward-targeted patch application.
- **`upstream/master` moves.** Lock base SHA `e9fa0781f1` at the start. (Current upstream base in tree: `876a4321`, fetched + re-rebased 2026-07-31.)
- **Remote parity drift.** A clean tree on `dspark` or any local branch is not proof of remote parity. Confirm with `rev-parse` / `ls-remote` SHA equality before declaring sync.
- **RESOLVED — work committed.** The 170-file uncommitted-tree risk is gone: everything was committed in 7 logical commits on `feature/turboquant-kv-cache-rebase` (tree clean, 2026-07-31). Remaining risks are runtime validation only.
- **NEW — runtime-untested.** Build/smoke tests pass, but no real-model inference, no turbo-KV-cache decode, no draft-model run has been executed yet.
- **RESOLVED — DSPARK is upstream-merged code.** The prior risk (DSPARK dropped on sync) is resolved: commit `84075273c` "spec: add DSpark speculative decoding (#25173)" (2026-07-28) is an ancestor of `upstream/master` (verified via `git merge-base --is-ancestor` against ref tip `5f55650a`, 2026-07-30), so DSPARK is merged upstream and is kept in this tree. `DGX-Spark` (NVIDIA hardware support) is separate and retained.
