# CUDA MoE expert cache

The CUDA MoE expert cache accelerates decode when routed expert weights remain in host memory. A cache hit runs the selected expert matvec on CUDA while the CPU computes the miss rows through the normal `MUL_MAT_ID` kernel. The cache belongs to one backend scheduler and persists until that scheduler is destroyed.

This is an opportunistic path. Unsupported nodes, unavailable cache capacity, contention, and cache failures fall back to CPU execution.

## Configuration

Use `--moe-cache MODE` with programs that use the common argument parser:

| Mode | Cache budget | Weight repacking | Device requirements |
| --- | --- | --- | --- |
| `auto` | Free VRAM minus the reserve | Preserved unless cache-aware fit selects canonical CPU experts | At least two eligible selected CUDA devices, compute capability 8.0 or newer |
| `on` | Free VRAM minus the reserve | Disabled | At least one eligible selected CUDA device, compute capability 7.0 or newer |
| `N` | At most `N` MiB per device, after the reserve | Disabled | Same as `on` |
| `off` or `0` | No cache session | Preserved unless changed separately | None |

`auto` is the default. When the complete model fits, or dense weights do not fit without routed experts, normal placement and repacking are preserved. When routed experts must spill and a cache is feasible, cache-aware fit keeps canonical expert weights in host memory, puts all dense layers on the main GPU when possible, and preserves the remaining VRAM for cache pools. Use `on` or a positive budget to force canonical CPU weights when automatic placement is not wanted.

The cache only sees experts assigned to host memory. Cache-aware fit uses a no-allocation model load to inventory exact expert shapes, aliases, model memory, context memory, and compute memory before choosing between stock and cache placement. Explicit `--cpu-moe`, `--n-cpu-moe`, tensor overrides, GPU layers, or tensor splits remain authoritative and can prevent fit from changing placement.

The cache considers only CUDA backends selected for the scheduler. It does not discover or use an unselected device.

## Eligibility and allocation

With default settings, a node must satisfy all of these conditions:

- The operation is the regular CPU `MUL_MAT_ID` path with F32 activations.
- The weight tensor name contains `_exps`.
- The weight type is supported by the CUDA quantized matvec kernel.
- In `auto`, one expert is at least 64 KiB; in `on` or fixed-budget mode, the default threshold is 1 MiB.
- The graph node contains no more than the configured maximum token batch and no more than 64 routed rows. The default maximum is eight tokens in `auto` and one token in forced modes.
- The selected device can hold a pool of at least 64 experts of that shape.

Each physical device budget is coordinated across every target, draft, MTP, and server scheduler in the process. It is claimed once, at first eligible use. Shape inventory counts the exact number of experts in every observed tensor; mixed expert counts do not reserve nonexistent slots. Device reservations are tracked per participating session, so releasing a high-reserve target immediately restores the correct capacity for the remaining sessions. The automatic budget is:

```
min(configured budget, free VRAM - 3072 MiB reserve)
```

The configured-budget term is omitted for `auto` and `on`. A fixed `N` is a cap for cache slabs and device-side dispatch scratch together, not a guaranteed slab allocation. Pool allocation may be smaller after reserving scratch or retrying an allocation, and no pool is created when fewer than 64 slots fit.

Tensor shapes are collected before pools are allocated. Allocation waits for a repeated shape census and 64 stable visits so early graph discovery does not give all capacity to the first tensor shape. Capacity is divided among discovered shapes. A layer is assigned once to a selected device and keeps that assignment while the device remains usable. If that device cannot host a different tensor shape from the layer, the tensor receives its own stable assignment. Initial assignments are deterministic and weighted by usable slab capacity after existing pools and dispatch scratch are accounted for.

When every routed row is resident in `auto`, one row stays on CPU while CUDA computes the other rows. This restores CPU/GPU overlap on a fully warm node. It improved DeepSeek V4 Flash by about 0.7% and GLM-5.2 by about 1.7% on the measured 4x RTX 3090 system. Nodes with an actual miss are unchanged.

The `[moe-cache] enabled` message is printed only after the first pool is allocated. If it is absent, the cache did not become active.

## Demand fill and eviction

The cache never transfers a missing expert synchronously for the current matvec. The miss stays on the CPU, and a bounded background request may populate the expert for a later use.

By default:

- An expert is admitted after its second observed miss.
- At most 8 fills are enqueued by one node.
- A device queue is limited to 128 jobs and 512 MiB.
- Each device has one low-priority CUDA fill stream.
- Host-to-device fills are serialized across devices in one session.
- Full pools use LRU eviction. After a successful fill, that expert needs eight fresh misses before it can replace another entry.

The default automatic maximum is eight tokens. Larger prompt-processing nodes bypass the cache, while single-token decode and current speculative verification batches remain eligible. Forced `on` and fixed-budget modes retain a conservative one-token default unless the environment control is changed.

## DeepSeek V4 Flash and MXFP4

The DeepSeek V4 Flash GGUF uses the conventional `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` tensors. These names contain `_exps`, and CPU-resident canonical weights reach the regular `MUL_MAT_ID` path, so the MXFP4 experts are compatible with the cache. For this model, the gate and up projections have `(n_in, n_out) = (4096, 2048)` and the down projection has `(n_in, n_out) = (2048, 4096)`. Each projection occupies 4,456,448 bytes per expert.

Automatic mode accepts up to eight tokens per node. This covers the current DSpark target-verification batch, whose tested node shape is six tokens selecting six experts each. Prompt nodes above eight tokens bypass the cache and do not drive admission. The model-free cache test compares the 6 x 6 hit path with the stock CPU result.

The server understands the target/draft dependency and is the supported DSpark entry point. With automatic fit, draft-device selection, and cache policy, a four-GPU launch is:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server \
    -m /path/to/DeepSeek-V4-Flash-0731.gguf \
    -md /path/to/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
    --spec-type draft-dspark --spec-draft-n-max 5 \
    --fit on --moe-cache auto \
    -t 24 -tb 48 -fa on -c 8192 -np 1 \
    -ctk q8_0 -ctv q8_0
```

When no draft device is explicit, the server places the DSpark sidecar on the eligible GPU with the most available memory, excluding the target main device when possible. The draft device is moved to the front of its own device list and receives its one-layer split while the remaining target backends stay available. The server measures this draft placement against both the packed and main-device no-allocation target placements, then reserves the per-device maximum before fitting. An explicit `--spec-draft-device` remains authoritative for draft weights, but target backends are appended when needed because DFlash and DSpark share the target embedding and output tensors. `--spec-draft-device none` keeps the sidecar weights on CPU while retaining those target backends for the shared operations. Sleep/wake reload starts from the original placement request and repeats measurement and fitting instead of accumulating old reservations.

On the measured 4x RTX 3090 system, the final target-only automatic policy reached `27.91 +/- 0.06` t/s versus `23.20 +/- 0.05` t/s with the cache off, a 20.3% end-to-end gain on the rebased implementation. Warm cache statistics were about 91-92% hits across the four devices with no fill, dispatch, or collect failures. A deterministic sequence with 100% draft acceptance reached 66.64 t/s with DSpark `n_max=5`, versus 26.93 t/s target-only. The same workload reached 58.90 t/s at `n_max=3`; `n_max=7` produced no additional extraction and reached 66.18 t/s. After excluding the target main device and reserving both possible shared-head placements, a repeated warm run reached 66.73 t/s, within the observed run variation, with 84.0-92.8% cache hits and no fill, dispatch, or collect failures. Explicit GPU, CPU-only draft, and sleep/wake reload paths also produced correct output. A low-acceptance prose request remained coherent at about 27.7 t/s, illustrating that speculative gain depends on the workload rather than the cache hit rate alone. Retain the 3072 MiB reserve or remeasure when increasing context, slots, or other CUDA allocations.

### Measured DeepSeek V4 validation

The following canonical-weight scaling sweep used the Q8_K_XL target, CPU-resident experts, all dense weights on the first RTX 3090, 256 untimed generation warmup tokens, and otherwise matched cache-off and cache-on arms:

| CUDA devices | Cache off | Cache on | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 19.05 t/s | 22.54 t/s | +18.3% |
| 2 | 18.95 t/s | 26.21 t/s | +38.3% |
| 3 | 18.86 t/s | 27.47 t/s | +45.6% |
| 4 | 18.59 t/s | 27.51 t/s | +48.0% |

Putting all dense weights on the first GPU remained faster than an equal four-device tensor split. A later fixed-budget sweep with that topology reached 23.91, 25.53, 26.45, 26.92, and 27.01 t/s at 4, 8, 12, 16, and 20 GiB per device respectively. The small change above 16 GiB shows that automatic capacity is already near the useful ceiling on this model. Disabling serialized fills did not improve steady decode, so serialized fills remain the default.

Quality checks used the actual target GGUF, not only synthetic tensors. Perplexity over four 512-token chunks was `2.7996 +/- 0.19833` with the cache off and `2.7987 +/- 0.19824` with it on. On the final accounting code, a 54,044-token retrieval prompt returned all three exact codes placed near the beginning, middle, and end. A same-session follow-up evaluated only 18 new prompt tokens, returned the correct middle code, and remained coherent. These results show statistical quality equivalence on the tested workload, not bit-identical execution.

Measure this model with three separate `llama-bench` arms and otherwise identical placement, context, batch, thread, and warmup settings:

1. `--moe-cache off --repack on`: optimized CPU-expert baseline.
2. `--moe-cache off --repack off`: canonical CPU-expert baseline.
3. `--moe-cache on --repack off`, or a fixed cache budget with `--repack off`: cache arm.

Other programs that use the common argument parser spell the repacking switches as `--repack` and `--no-repack` rather than `--repack on` and `--repack off`.

Comparing arm 1 with arm 3 measures the end-to-end configuration benefit; comparing arm 2 with arm 3 isolates cache execution from repacking. A cache result is valid only when the logs contain `[moe-cache] enabled`, the expected MXFP4 pools, and nonzero hits in periodic or teardown statistics. Before reporting a speedup, also compare cache off and on for perplexity or logits, coherent generation, and retrieval or prompt-following at increasing context lengths. Token streams need not be identical because CPU and CUDA rounding can change near ties.

## GLM-5.2 validation and production profile

The GLM-5.2 UD-IQ2_M tests used four RTX 3090 GPUs and the same 48-core host. On the final rebased implementation, the complete automatic policy improved from `14.49 +/- 0.04` t/s with the cache off to `19.45 +/- 0.23` t/s in auto, or 34.2%. The off arm preserved repacking; auto selected canonical experts plus about 62,981 MiB of projected cache capacity in the active arm. The cache needs a longer warmup than DeepSeek because the routed expert set is about 209 GiB; short runs can continue accelerating while fills complete.

The model contains one embedded next-token prediction layer. It works with the cache, but the best measured settings were `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5`, which reached 20.62 t/s on that workload, 18.6% above the target-only server. Increasing the draft maximum to five reduced throughput to about 15.9 t/s because acceptance fell while draft work increased.

A 58,759-token archival prompt returned all three exact retrieval markers. Its 219-token answer ran at 13.38 t/s with 159 of 173 draft tokens accepted. A same-session follow-up reused 58,769 prompt tokens, correctly classified the early, middle, and late records, and produced 294 coherent tokens at 12.84 t/s with 210 of 230 draft tokens accepted. Perplexity over four 512-token chunks was `2.2090 +/- 0.11977` with the cache off and `2.1960 +/- 0.11996` with it on.

Embedded MTP creates another context, scheduler, and cache session. Sessions retain independent cache state, but their claims are coordinated per physical GPU so target and draft contexts do not each budget the same free memory. In the long-context run, allocator pressure correctly trimmed a cache pool and the request still completed exactly. This fallback is safe, but repeated trim is undesirable for production.

The validated three-GPU production profile therefore uses target-only decoding:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 \
GGML_CUDA_MOE_CACHE_RESERVE_MB=3072 \
./build/bin/llama-server \
    -m /path/to/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf \
    --fit off -ngl 99 -ncmoe 99 \
    --moe-cache on --no-repack -sm layer -ts 1,0,0 \
    -c 65536 -np 1 -b 4096 -ub 512 \
    -t 44 -tb 48 -fa on --load-mode mmap --cache-ram 0
```

At context 65,536, a 2048-token ubatch required about 3952 MiB for the prompt graph and failed to fit; 512 was stable. The first GPU held the dense weights and context and had about 1.5 GiB free, so its cache stayed dormant under the 3 GiB reserve. The other two GPUs supplied the expert cache. On the final binary, two coherent generations reached 15.54 and 16.46 t/s while physical GPU 3 remained at 1 MiB. After lazy pool growth, GPUs 1 and 2 used about 21 GiB each.

`--load-mode none` was also tested against mmap with the exact 65k server profile. No-mmap took 4:31 to become ready instead of 3:03, left only about 28 GiB host memory available, increased swap use, and averaged 14.88 t/s over two steady samples versus 14.96 t/s for mmap. Mmap therefore remains the tested default for this profile.

## Broader regression matrix

These matched canonical-weight decode checks force routed experts to CPU memory. Small models used one RTX 3090; models that required or benefited from more capacity used all four. They exercise different expert counts, quantizations, and architectures:

| Model | Cache off | Cache on | Change |
| --- | ---: | ---: | ---: |
| OLMoE-1B-7B Q4_K_M | 200.52 t/s | 254.12 t/s | +26.7% |
| ERNIE-4.5-21B-A3B Q4_K_M | 79.66 t/s | 127.40 t/s | +59.9% |
| Qwen3-30B-A3B Q4_K_XL | 71.93 t/s | 73.43 t/s | +2.1% |
| Qwen3.6-35B-A3B Q4_K_XL | 82.32 t/s | 82.09 t/s | -0.3% |
| DeepSeek-V2-Lite Q4_K_M | 65.04 t/s | 81.61 t/s | +25.5% |
| Llama-4 Scout Q4_K_XL | 25.88 t/s | 45.03 t/s | +74.0% |
| MiniMax M2.7 IQ2_XXS | 37.76 t/s | 47.81 t/s | +26.6% |

The Qwen3.6 experts are below the default 1 MiB admission threshold, so the cache remained effectively dormant and the result is parity rather than an active-cache regression. Llama-4 used 512 warmup and generation tokens; its cache result was 45.03 +/- 0.24 t/s after the shorter warmup showed fill variance. This matrix is evidence for the tested hardware, not a guarantee that every eligible GPU and model will improve.

The real OLMoE model also exercised every available quant family from Q2_K through Q8_0 with forced 64 KiB admission. Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_0, Q5_K_S/M, Q6_K, and Q8_0 all engaged the cache with no fill, dispatch, or collect failures. F16 remained dormant because that cache matvec type is unsupported. A fully resident Qwen3.6-35B control preserved repacking and remained dormant at 146.74 t/s off versus 146.65 t/s auto.

## Concurrency, fallback, and lifetime

The scheduler binds its cache session only while it computes a graph. Within one session, per-device scratch buffers and dispatch are serialized. If another cached node in that session already owns a device, the contending node takes the regular CPU path. Separate schedulers own separate sessions, pools, streams, and dispatch locks even when they select the same physical device.

A two-slot DeepSeek server test generated unrelated Saturn and photosynthesis answers concurrently. Both responses were complete, coherent, and isolated, with no deadlock, row leakage, or cache failure. This complements the model-free concurrent-node and nested-session tests.

Valid slots are pinned for the lifetime of a node. Miss rows remain in the normal CPU work set. Hit rows are removed from that set only after the complete GPU dispatch has been accepted.

If dispatch fails, every planned hit row is restored to the CPU work set before worker threads begin. If collection fails, every skipped row is recomputed with the stock CPU helper. A fatal CUDA cache error disables and trims the affected device, after which nodes continue on CPU.

CUDA output can differ slightly from CPU output because the hit path uses CUDA activation quantization and matvec arithmetic. Do not expect bit-identical logits or token streams. In particular, a small rounding difference can change a near-tie greedy token.

Public writes to a cached host weight buffer invalidate the affected byte range before the write begins. Invalidation cancels overlapping queued fills, waits for overlapping active reads and transfers, and removes overlapping slots and demand records. The same process runs before a host allocation is released or stops being a weight buffer. Callers must still obey the normal backend synchronization rules when mutating a weight used by concurrent graph execution. Scheduler teardown stops admission, cancels queued work, waits for active graph scopes and nodes, joins fill workers, and then frees device storage.

The normal CUDA allocator may trim an active expert cache as a last attempt to satisfy an allocation. Trimming frees all cache storage on that device and leaves it disabled for the rest of the session. Device scratch growth also retries once after releasing the superseded scratch allocation when replacement overlap causes an out-of-memory result. A session becomes permanently dormant when no nonzero device budget remains, or when `auto` drops below two devices with nonzero budgets; any remaining cache devices are then trimmed as well.

## Diagnostics and developer controls

The pool log reports the physical CUDA device, weight type, expert size, slot count, and allocated bytes. Session teardown reports hits, misses, queue activity, fills, evictions, CPU-overlap rows, activation deduplication, and fallback counters for devices that processed cache nodes. Set `GGML_CUDA_MOE_CACHE_STATS=N` to print the same counters every `N` collection calls.

The following environment variables are implementation controls, not a stable command-line interface. They are read when a scheduler creates its cache session.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `GGML_CUDA_MOE_CACHE_RESERVE_MB` | `3072` | VRAM left outside the cache on each device |
| `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB` | mode dependent | Minimum bytes per expert, in KiB; `64` in auto and `1024` when forced |
| `GGML_CUDA_MOE_CACHE_MAX_BATCH` | mode dependent | Maximum tokens in an eligible node; `8` in auto and `1` when forced |
| `GGML_CUDA_MOE_CACHE_INSERTS` | `8` | Maximum admissions per node |
| `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` | `2` | Miss count required before admission |
| `GGML_CUDA_MOE_CACHE_THROTTLE` | `8` | Fresh misses required before replacing a full-pool entry |
| `GGML_CUDA_MOE_CACHE_QUEUE` | `128` | Maximum queued jobs per device |
| `GGML_CUDA_MOE_CACHE_QUEUE_MB` | `512` | Maximum queued source bytes per device |
| `GGML_CUDA_MOE_CACHE_STATS` | `0` | Collection-call interval for periodic statistics, or `0` for teardown only |
| `GGML_CUDA_MOE_CACHE_NDEV` | all | Maximum selected CUDA devices used by a session |
| `GGML_CUDA_MOE_CACHE_SERIAL_FILL` | `1` | Serialize fills across devices in a session |
| `GGML_CUDA_MOE_CACHE_DEDICATED_MMV` | `1` | Use the cache-specific matvec activation map; `0` enables the compatible modulo-index fallback for comparison |
| `GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS` | mode dependent | Rows retained on CPU only when every row is resident; `1` in auto and `0` when forced |
| `GGML_CUDA_MOE_CACHE_MIN_CC` | mode dependent | Override the minimum compute capability encoded as `major * 100 + minor * 10` |

Directly setting `GGML_CUDA_MOE_CACHE_MODE`, `GGML_CUDA_MOE_CACHE`, or `GGML_CUDA_MOE_CACHE_BUDGET_MB` controls the provider only when a program leaves the mode unspecified. Common applications do this for their implicit `auto` default, so provider settings affect fit and runtime consistently. An explicit `--moe-cache` or `LLAMA_ARG_MOE_CACHE` value takes precedence and also controls the model loader's repacking choice. `llama-bench` applies its selected cache arm before every model instance and overwrites the three raw backend variables; use `--moe-cache` or `LLAMA_ARG_MOE_CACHE` to select its benchmark mode.

Keep `GGML_OP_OFFLOAD_MIN_BATCH` above the decode batch size. Setting it to `1` can make the scheduler offload the complete `MUL_MAT_ID` operation to CUDA before the CPU path can split cache hits from misses.

`GGML_CUDA_MOE_CACHE_FAIL` is for fallback testing only. It accepts `dispatch`, `collect`, `insert`, or `slab`; comma-separated stages and `all` are also accepted. A CUDA build can exercise the synthetic success and failure paths with:

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/test-moe-cache
```

## Benchmarking

Cache measurements need decode warmup. Pool creation waits for graph-shape discovery, expert admission needs repeated demand, and queued fills finish asynchronously. A one-token warmup usually measures a cold cache rather than steady state.

`llama-bench` separates devices within one tensor split with `/`, for example `-ts 1/0/0/0`; commas separate multiple benchmark values. Programs using the common parser, including `llama-server`, use commas within one split, for example `-ts 1,0,0,0`.

For an end-to-end comparison of automatic policy, let fit choose each arm and vary only `off` versus `auto`:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-bench \
    -m /path/to/model.gguf -p 0 -n 300 --n-gen-warmup 256 -r 5 \
    -ngl 99 -fitt 1024 -t CPU_THREAD_COUNT -fa on \
    --moe-cache off,auto -o json
```

Replace the uppercase placeholder with an integer appropriate for the host. This measures the complete automatic choice, which may use different expert placement in the two arms. Check the logs to confirm that the `auto` arm selected cache placement and allocated pools. Record GPU models, PCIe topology, CPU model, memory channels and speed, model quantization, exact placement, build revision, and all environment overrides.

For an isolated comparison with the same canonical CPU weights in both arms, use a fixed cache budget and explicitly disable repacking:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 ./build/bin/llama-bench \
    -m /path/to/model.gguf -p 0 -n 300 --n-gen-warmup 256 -r 5 \
    -ngl 99 -ncmoe MODEL_MOE_LAYER_COUNT -t CPU_THREAD_COUNT -fa on \
    -sm layer -ts 1/0/0 --moe-cache off,4096 --repack off -o json
```

Adjust `4096` to the intended per-device MiB cap. A layer split with trailing zero shares keeps dense work on the first GPU while still creating every selected CUDA backend for cache use. `-sm none` creates only the main CUDA backend and is not a valid multi-GPU cache-capacity comparison. `llama-bench` records the effective repack setting and rejects repacking with cache `on` or a fixed budget.

Use a long enough timed generation to amortize graph discovery and inspect both throughput variation and the final cache counters. Repeat the process after a cold process start when cold-start behavior matters. Do not infer a gain from hit rate alone: activation transfers, result transfers, fill traffic, GPU speed, PCIe speed, and CPU memory bandwidth all affect the result.

An `off` versus `on` comparison without `--repack off` includes the intended repacking-policy change. Treat results that use different repacking or expert placement as an end-to-end configuration comparison, not an isolated measurement of the cache.

## Current limitations

- CUDA only. HIP, MUSA, Metal, Vulkan, and other backends do not register an implementation.
- CPU-resident expert `MUL_MAT_ID` only. There is no fused gate/up/activation path and no GPU-resident output handoff.
- Demand fill only. There is no predictive prefetch or separate prompt-time population path.
- No hot-set file or persistence across scheduler sessions or process restarts.
- Direct writes through a raw host pointer bypass invalidation. Mutate cached weight buffers through the backend tensor and buffer APIs.
- No runtime performance bail-out. An eligible but unprofitable cache remains active unless it fails, is trimmed, or is disabled by configuration.
- Every scheduler owns independent residency state. Process-wide physical-device coordination prevents those sessions from claiming the same free VRAM, but it does not share expert slots between them.
- Unloading a dynamic CUDA backend while a scheduler or cache session created by it is still alive is unsupported. Destroy those objects before unloading the backend module.
- Generic operation offload can bypass the cache if `GGML_OP_OFFLOAD_MIN_BATCH` is set low enough to offload decode nodes.
- Performance depends strongly on model shape, quantization, CPU memory bandwidth, PCIe link, CUDA device, spare VRAM, and routing locality.
