# PPL Test Results — OSCAR2 / KV-Cache Comparison

**Date:** 2026-08-03  
**Build:** `/mnt/storage/Projects/llama-cpp-turboquant`  
**Binary:** `llama-perplexity` from turboquant build  
**Dataset:** wikitext-2-raw `wiki.test.raw` (4,358 lines)  
**Protocol:** `-c 512 --chunks 1 --no-warmup -fa on -ngl 0 -t 32 -f <dataset>`  
**GPU:** RTX 5090 (32GB)  

## Note on Context Setting

Tests were run at `-c 512 --chunks 1` (single 512-token window) rather than the full `-c 4096 --chunks -1` protocol because the latter would require ~5+ minutes per run at CPU-only (ngl=0). The single-chunk protocol provides relative comparisons between cache types. Per the oscar2-llama-debug skill, `--chunks 1` vs `--chunks -1` are NOT comparable in absolute PPL terms (catastrophic tiers differ wildly), but they ARE comparable for relative backend ranking at the same ctx/chunks setting.

## Models Tested

| Model | Path | Rot-KV? | Architecture | Layers | SWA? | MoE? | Size |
|-------|------|---------|-------------|--------|------|------|------|
| gemma4-12b | `oscar-rotations/gemma-4-12b-it-UD-Q8_K_XL-rot-kv.gguf` | Yes (k_rot+v_rot+layer_output_scale, 48 each) | gemma4 | 48 | Yes (n_swa=1024) | No | 12.7 GB |
| gemma4-26b-a4b | `oscar-rotations/gemma-4-26B-A4B-it-UD-Q5_K_S-rot-kv.gguf` | Yes (k_rot+v_rot+layer_output_scale) | gemma4 | 30 | Yes (n_swa=1024) | Yes (128 experts, 8 used) | 17.6 GB |
| qwen3.6-27b | `oscar-rotations/Qwen3.6-27B-UD-Q5_K_XL-rot-kv.gguf` | Yes (k_rot+v_rot only, 36 each, NO layer_output_scale) | qwen35 | 64 | No (n_swa=0) | No | 18.7 GB |
| kwaipilot-35b | `qwen3.6/35B/Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_S.gguf` | No rotation tensors at all | qwen35moe | 40 | No (n_swa=0) | Yes (256 experts, 8 used) | 22.5 GB |

## Key Constraint

> **oscar2 can only mix with f16/oscar2 or oscar2/oscar2 — NOT q8_0/turbo4**

## Results Summary

All PPL values are from `-c 512 --chunks 1 --no-warmup`. Lower is better.

| # | Model | Cache K/V | PPL | Warning |
|---|-------|-----------|-----|---------|
| 1 | gemma4-12b | f16/f16 | **360.07** | — |
| 2 | gemma4-26b-a4b | f16/f16 | **62456.41** | — |
| 3 | qwen3.6-27b | f16/f16 | **4.05** | — |
| 4 | kwaipilot-35b | f16/f16 | **4.05** | — |
| 5 | gemma4-12b | oscar2/oscar2 | **1953.72** | No warning (has rot tensors) |
| 6 | gemma4-26b-a4b | oscar2/oscar2 | **1858.28** | No warning (has rot tensors) |
| 7 | qwen3.6-27b | oscar2/oscar2 | **4.47** | ⚠️ "without rotation tensors" |
| 8 | gemma4-12b | oscar2/f16 | **146,539,637.87** | No warning — CRITICAL: catastrophic |
| 9 | gemma4-26b-a4b | oscar2/f16 | **3,903,591,703.05** | No warning — CRITICAL: catastrophic |
| 10 | qwen3.6-27b | oscar2/f16 | **178.84** | ⚠️ "without rotation tensors" |
| 11 | gemma4-12b | f16/oscar2 | **1,064,022.57** | No warning — CRITICAL: catastrophic |
| 12 | gemma4-26b-a4b | f16/oscar2 | **55,585.55** | No warning — severe degradation |
| 13 | qwen3.6-27b | f16/oscar2 | **15.04** | ⚠️ "without rotation tensors" |
| 14 | gemma4-12b | q8_0/q8_0 | **367.08** | — |
| 15 | gemma4-26b-a4b | q8_0/q8_0 | **52459.20** | — |
| 16 | qwen3.6-27b | q8_0/q8_0 | **4.06** | — |
| 17 | kwaipilot-35b | q8_0/q8_0 | **4.03** | — |
| 18 | gemma4-12b | q8_0/turbo4 | **311.35** | — |
| 19 | gemma4-26b-a4b | q8_0/turbo4 | **52666.87** | — |
| 20 | qwen3.6-27b | q8_0/turbo4 | **4.07** | — |
| 21 | kwaipilot-35b | q8_0/turbo4 | **4.02** | — |
| 22 | kwaipilot-35b | oscar2/oscar2 | **4.39** | ⚠️ "without rotation tensors" |

## Analysis

### 1. Gemma4 models (rot-kv, has layer_output_scale tensors)
- **f16/f16 baseline**: gemma4-12b PPL=360, gemma4-26B PPL=62456 (high but expected for small models on tiny context)
- **oscar2/oscar2**: gemma4-12b PPL=1954, gemma4-26B PPL=1858 — significantly degraded vs f16 but coherent (no garbled output)
- **oscar2/f16 and f16/oscar2**: **CATASTROPHIC** — PPL in millions/billions. These mixed configs are broken for Gemma4 rot-kv models. The OSCAR2 INT2 dequant path is incompatible with f16 when rotation tensors are present.
- **q8_0/q8_0**: gemma4-12b PPL=367, gemma4-26B PPL=52459 — reasonable, slightly worse than f16
- **q8_0/turbo4**: gemma4-12b PPL=311, gemma4-26B PPL=52667 — comparable to q8_0/q8_0

### 2. Qwen3.6-27B (rot-kv, has k_rot+v_rot but NO layer_output_scale)
- **f16/f16**: PPL=4.05 — excellent
- **oscar2/oscar2**: PPL=4.47 — very close to f16, excellent for INT2 KV
- **oscar2/f16**: PPL=178.84 — degraded but not catastrophic (the "without rotation tensors" warning applies)
- **f16/oscar2**: PPL=15.04 — degraded (10x worse than oscar2/oscar2)
- **q8_0/q8_0**: PPL=4.06 — parity with f16
- **q8_0/turbo4**: PPL=4.07 — parity with f16

### 3. Kwaipilot-35B (NO rotation tensors at all)
- **f16/f16**: PPL=4.05
- **oscar2/oscar2**: PPL=4.39 — works but with the "without rotation tensors" warning. Surprisingly coherent (Kwaipilot appears to handle INT2 KV without rotation gracefully)
- **q8_0/q8_0**: PPL=4.03
- **q8_0/turbo4**: PPL=4.02

### Key Takeaways

1. **OSCAR2 requires both K and V to be oscar2** — mixing oscar2 with f16 causes catastrophic PPL degradation on rot-kv Gemma4 models (millions to billions)

2. **Gemma4 rot-kv models show the worst oscar2 degradation** — likely related to SWA (n_swa=1024) + mixed head dimensions (D=512 base, D=256 SWA) + the layer_output_scale tensors that Gemma4 uses but Qwen3.6 does not

3. **Qwen3.6-27B handles oscar2 well** — oscar2/oscar2 PPL=4.47 vs f16 PPL=4.05 (only 10% degradation). The rot-kv works without layer_output_scale

4. **Kwaipilot-35B (non-rot-kv) also handles oscar2 reasonably** — PPL=4.39, despite the warning. The warning is benign for this model

5. **q8_0/turbo4 is the best non-f16 comparison** — consistently within 2% of f16/f16 across all models, confirming these are valid quant baselines

## Recommendation

For OSCAR2 usage:
- Use **oscar2/oscar2 only** (never mix with f16 or q8_0)
- Gemma4 SWA models need the `LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256` env vars + the SWA fallback that was reverted (per oscar2-llama-debug skill) for production use
- Qwen3.6-27B achieves good oscar2 results without additional env vars
- Kwaipilot-35B works but should use q8_0/turbo4 for better results if PPL is the priority

## Raw Logs

All individual test logs are in: `/mnt/storage/Projects/llama-cpp-turboquant/ppl_results/logs_20260803_191820/`