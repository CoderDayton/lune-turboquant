#include "common.h"
#include "dequantize.h"

// ============================================================================
// TurboFlash: Two-pass fused asymmetric attention (K=q8_0, V=turbo3)
// ============================================================================
//
// Eric Kryski's TurboFlash architecture, ported from the MLX implementation.
//
// Pass 1: Each threadgroup processes a block of B=64 KV tokens in parallel.
//   - Computes Q·K scores for the block (K read from q8_0 via dequant)
//   - Online softmax within the block (running max + exp sum)
//   - Weighted V sum (turbo3 dequant: 3-bit centroid lookup with norm scaling)
//   - Stores per-block partial results: {partial_output[dim], block_max, block_sum}
//
// Pass 2: One threadgroup per query head merges all block partial results.
//   - Online softmax merge across blocks (combine per-block max/sum -> global)
//   - Rescale partial outputs by correction factor
//   - Sum into final output
//   - Apply inverse WHT rotation (signs2 -> butterfly -> signs1)
//
// Why two-pass beats single-pass: the single-pass kernel assigns one threadgroup
// per query head, serializing ALL T_kv tokens. With two-pass at B=64:
//   Pass 1: T/64 threadgroups run in parallel (massive GPU occupancy)
//   Pass 2: 1 threadgroup merges T/64 partial results (trivially fast)
//
// Benchmarked B=32, B=64, B=128 — B=64 is optimal for Apple Silicon.

// V4 architecture: 32 threads (1 SIMD group) per block, tokens processed in a loop.
// Tokens per block stays 64 for pass 2 compatibility.
#define TURBO_FLASH_BLOCK_SIZE 64
#define TURBO_FLASH_TG_SIZE    32

// Function constants for Pass 1
constant int32_t FC_turbo_flash_p1_dk  [[function_constant(FC_TURBO_FLASH_P1 + 0)]];
constant int32_t FC_turbo_flash_p1_dv  [[function_constant(FC_TURBO_FLASH_P1 + 1)]];
constant bool    FC_turbo_flash_p1_has_mask [[function_constant(FC_TURBO_FLASH_P1 + 2)]];
constant bool    FC_turbo_flash_p1_k_is_turbo3 [[function_constant(FC_TURBO_FLASH_P1 + 3)]];

// Function constants for Pass 2
constant int32_t FC_turbo_flash_p2_dv  [[function_constant(FC_TURBO_FLASH_P2 + 0)]];

// ---------------------------------------------------------------------------
// Pass 1 V4: SIMD-per-token architecture (Eric Kryski pattern)
// ---------------------------------------------------------------------------
//
// Grid: (n_bh, n_blocks, 1)
//   n_bh = ne01 * ne02 * ne03 (total query heads across batch)
//   n_blocks = ceil(T_kv / TURBO_FLASH_BLOCK_SIZE)
//
// Threadgroup: (32, 1, 1) — exactly 1 SIMD group
//
// Each SIMD group of 32 threads processes one block of 64 KV tokens for one
// query head. Each lane handles DIMS_PER_LANE = DK/32 interleaved dimensions
// (e.g., lane 0 → dims 0,32,64,96 for DK=128).
//
// Key differences from V3:
//   - NO shared memory — Q, codebook, softmax state, V accum all in registers
//   - K scoring via per-lane partial dot product + simd_sum reduction
//   - Online softmax across tokens (not across threads within a token)
//   - V accumulation fused with softmax in the token loop
//
// Buffer layout (unchanged from V3 for pass 2 compatibility):
//   partial_out: float[n_bh * n_blocks * dv]  — per-block weighted V accumulators
//   partial_ms:  float[n_bh * n_blocks * 2]   — per-block {max_score, exp_sum}
//
template <short DK, short DV>
kernel void kernel_turbo_flash_p1(
        constant ggml_metal_kargs_turbo_flash_p1 & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       float * partial_out,
        device       float * partial_ms,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]]) {

    constexpr short DK_PER_LANE = DK / 32;
    constexpr short DV_PER_LANE = DV / 32;

    const uint lane     = tiitg % 32;  // SIMD lane (0-31)
    const uint bh_idx   = tgpig[0];
    const uint block_id = tgpig[1];

    const int T_kv      = args.ne11;
    const int n_blocks  = args.n_blocks;

    // Token range for this block
    const int t_start = (int)(block_id * TURBO_FLASH_BLOCK_SIZE);
    const int t_end   = min(t_start + TURBO_FLASH_BLOCK_SIZE, T_kv);

    // Decompose bh_idx back to (iq1, iq2, iq3) for pointer arithmetic
    const uint iq1 = bh_idx % args.ne01;
    const uint iq2 = (bh_idx / args.ne01) % args.ne02;
    const uint iq3 = bh_idx / (args.ne01 * args.ne02);

    // GQA: map query head to KV head
    const uint ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const uint ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    // ====== Load Q into registers ======
    // Each lane loads its interleaved dims: lane, lane+32, lane+64, ...
    device const float * q_ptr = (device const float *)((device const char *)q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);
    float q_vals[DK_PER_LANE];
    for (short i = 0; i < DK_PER_LANE; i++) {
        const int d = (int)lane + i * 32;
        q_vals[i] = (d < DK) ? q_ptr[d] : 0.0f;
    }

    // ====== Load turbo3 codebook into registers (8 entries) ======
    float v_cb[8];
    for (int i = 0; i < 8; i++) {
        v_cb[i] = float(turbo_centroids_3bit_h[i]);
    }

    // K codebook — same centroids, only loaded when K is turbo3
    float k_cb[8];
    if (FC_turbo_flash_p1_k_is_turbo3) {
        for (int i = 0; i < 8; i++) {
            k_cb[i] = float(turbo_centroids_3bit_h[i]);
        }
    }

    // ====== Online softmax state — all in registers ======
    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state[DV_PER_LANE];
    for (short i = 0; i < DV_PER_LANE; i++) {
        o_state[i] = 0.0f;
    }

    // ====== K/V base pointers for this KV head ======
    device const char * k_base = (device const char *)k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * v_base = (device const char *)v + ikv2*args.nb22 + ikv3*args.nb23;

    // Mask pointer (precompute once)
    device const half * mask_ptr = nullptr;
    if (FC_turbo_flash_p1_has_mask) {
        mask_ptr = (device const half *)(mask + iq1*args.nb31 + (iq2 % args.ne32)*args.nb32 + (iq3 % args.ne33)*args.nb33);
    }

    // ====== Process all tokens in this block ======
    for (int t = t_start; t < t_end; t++) {

        // --- Check mask ---
        float mask_val = 0.0f;
        if (FC_turbo_flash_p1_has_mask) {
            mask_val = (float)mask_ptr[t];
            if (mask_val <= -MAXHALF) {
                continue;  // masked out — skip entirely
            }
        }

        // --- Dequant K and compute Q·K score ---
        // Each lane computes partial dot for its interleaved dims, then simd_sum.
        float dot_partial = 0.0f;

        if (FC_turbo_flash_p1_k_is_turbo3) {
            // K is turbo3_0: same struct as V — norm, qs[], signs[]
            device const block_turbo3_0 * k_row = (device const block_turbo3_0 *)(k_base + t * args.nb11);
            const float k_norm = float(k_row[0].norm);

            for (short i = 0; i < DK_PER_LANE; i++) {
                const int d = (int)lane + i * 32;
                if (d >= DK) break;

                const int qs_byte = d / 4;
                const int qs_shift = (d % 4) * 2;
                const uint8_t q_idx = (k_row[0].qs[qs_byte] >> qs_shift) & 0x03;

                const int sign_byte = d / 8;
                const int sign_bit  = d % 8;
                const uint8_t s_bit = (k_row[0].signs[sign_byte] >> sign_bit) & 1;

                const uint8_t centroid_idx = q_idx | (s_bit << 2);
                dot_partial += q_vals[i] * k_cb[centroid_idx] * k_norm;
            }
        } else {
            // K is q8_0: 32 elements per block, DK/32 blocks per row.
            device const block_q8_0 * k_row = (device const block_q8_0 *)(k_base + t * args.nb11);

            for (short i = 0; i < DK_PER_LANE; i++) {
                const int d = (int)lane + i * 32;
                if (d >= DK) break;

                // Which q8_0 block and offset within it
                const int qb = d / 32;   // block index
                const int qj = d % 32;   // element within block
                dot_partial += q_vals[i] * (float)k_row[qb].qs[qj] * (float)k_row[qb].d;
            }
        }
        float score = simd_sum(dot_partial) * args.scale + mask_val;

        // --- Dequant V for this token ---
        // V is turbo3: one block_turbo3_0 per 128 dims (QK_TURBO3=128).
        // Each lane unpacks its interleaved dims.
        device const block_turbo3_0 * v_row = (device const block_turbo3_0 *)(v_base + t * args.nb21);
        const float v_norm = float(v_row[0].norm);

        float v_decoded[DV_PER_LANE];
        for (short i = 0; i < DV_PER_LANE; i++) {
            const int d = (int)lane + i * 32;
            if (d >= DV) { v_decoded[i] = 0.0f; continue; }

            const int qs_byte = d / 4;
            const int qs_shift = (d % 4) * 2;
            const uint8_t q_idx = (v_row[0].qs[qs_byte] >> qs_shift) & 0x03;

            const int sign_byte = d / 8;
            const int sign_bit  = d % 8;
            const uint8_t s_bit = (v_row[0].signs[sign_byte] >> sign_bit) & 1;

            const uint8_t centroid_idx = q_idx | (s_bit << 2);
            v_decoded[i] = v_cb[centroid_idx] * v_norm;
        }

        // --- Online softmax update + V accumulation ---
        float new_m = max(m_state, score);
        float exp_diff  = exp(m_state - new_m);
        float exp_score = exp(score - new_m);

        for (short i = 0; i < DV_PER_LANE; i++) {
            o_state[i] = o_state[i] * exp_diff + exp_score * v_decoded[i];
        }

        l_state = l_state * exp_diff + exp_score;
        m_state = new_m;
    }

    // ====== Write partial results ======
    // Each lane writes its interleaved dims to partial_out (scatter write, no reduction)
    for (short i = 0; i < DV_PER_LANE; i++) {
        const int d = (int)lane + i * 32;
        if (d < DV) {
            partial_out[bh_idx * (uint64_t)n_blocks * DV + block_id * DV + d] = o_state[i];
        }
    }

    // Lane 0 writes block max and sum
    if (lane == 0) {
        partial_ms[bh_idx * (uint64_t)n_blocks * 2 + block_id * 2 + 0] = m_state;
        partial_ms[bh_idx * (uint64_t)n_blocks * 2 + block_id * 2 + 1] = l_state;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Merge block partial results + inverse WHT rotation
// ---------------------------------------------------------------------------
//
// Grid: (n_bh, 1, 1)
// Threadgroup: (tg_size, 1, 1) where tg_size >= DV (for WHT butterfly)
//
// Merges partial results from Pass 1 across all blocks using online softmax
// correction, then applies inverse WHT rotation (signs2 -> FWHT -> signs1).
//
template <short DV>
kernel void kernel_turbo_flash_p2(
        constant ggml_metal_kargs_turbo_flash_p2 & args,
        device const float * partial_out,
        device const float * partial_ms,
        device       float * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]]) {

    const uint tid    = tiitg;
    const uint bh_idx = tgpig[0];
    const int  n_blocks = args.n_blocks;

    // Shared memory for merge + WHT
    threadgroup float * shared_out = shmem;

    // ====== Step 1: Find global max across all blocks ======
    // Single thread scans — n_blocks is typically small (T/64)
    threadgroup float * shared_global_max = shmem + DV;
    threadgroup float * shared_global_sum = shmem + DV + 1;

    if (tid == 0) {
        float gmax = -INFINITY;
        for (int b = 0; b < n_blocks; b++) {
            gmax = max(gmax, partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 0]);
        }
        shared_global_max[0] = gmax;

        // Compute global exp sum with correction
        float gsum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float bmax = partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 0];
            float bsum = partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 1];
            float correction = exp(bmax - gmax);
            gsum += correction * bsum;
        }
        shared_global_sum[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = shared_global_max[0];
    float global_sum = shared_global_sum[0];
    float inv_global_sum = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    // ====== Step 2: Merge partial outputs with softmax correction ======
    if ((int)tid < DV) {
        float accum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float bmax = partial_ms[bh_idx * (uint64_t)n_blocks * 2 + b * 2 + 0];
            float correction = exp(bmax - global_max);
            float block_val = partial_out[bh_idx * (uint64_t)n_blocks * DV + b * DV + tid];
            accum += correction * block_val;
        }
        // Normalize by global softmax sum
        shared_out[tid] = accum * inv_global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====== Step 3: Inverse WHT rotation (signs2 -> FWHT -> signs1) ======
    // The turbo3 V cache stores values in WHT-rotated space.
    // We need to apply the inverse rotation to get back to the original space.
    // Inverse WHT: multiply by signs2, apply FWHT butterfly, multiply by signs1, scale by 1/sqrt(dim).

    // Step 3a: Apply signs2
    if ((int)tid < DV && (int)tid < 128) {
        shared_out[tid] *= turbo_wht_signs2[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3b: FWHT butterfly — simd_shuffle_xor for first 5 stages (zero barriers)
    // then shared memory for remaining stages (stages 5-6 for dim=128)
    // Ported from Eric Kryski's V2.1 optimization.
    if ((int)tid < DV && (int)tid < 128) {
        const int dim_wht = min((int)DV, 128);
        float val = shared_out[tid];

        // Phase 1: Intra-SIMD butterfly via simd_shuffle_xor (stages 0..4)
        // Register-to-register, no shared memory or barriers needed
        const uint lane_in_simd = tid % 32;
        // log2(dim_wht) for power-of-2 dims: 128->7, 64->6, 32->5
        const int log2_dim = (dim_wht >= 128) ? 7 : (dim_wht >= 64) ? 6 : 5;
        const int simd_stages = min(5, log2_dim);
        for (int s = 0; s < simd_stages; s++) {
            const uint step = 1u << s;
            float other = simd_shuffle_xor(val, (ushort)step);
            val = (lane_in_simd & step) ? (other - val) : (other + val);
        }

        // Phase 2: Cross-SIMD butterfly via shared memory (stages 5+)
        // Only needed for dim > 32
        if (dim_wht > 32) {
            shared_out[tid] = val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int half_block = 32; half_block < dim_wht; half_block <<= 1) {
                const int bfly_size = half_block << 1;
                const int bfly_idx = (int)tid / bfly_size;
                const int local_idx = (int)tid % bfly_size;
                const int base_idx = bfly_idx * bfly_size;

                float a = shared_out[base_idx + (local_idx % half_block)];
                float b_val_s = shared_out[base_idx + (local_idx % half_block) + half_block];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                shared_out[tid] = (local_idx < half_block) ? (a + b_val_s) : (a - b_val_s);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            val = shared_out[tid];
        }

        // Step 3c: Apply signs1 and normalize
        float inv_sqrt_dim = rsqrt((float)dim_wht);
        dst[bh_idx * DV + tid] = val * inv_sqrt_dim * turbo_wht_signs1[tid];
    }

    // For DV > 128: elements beyond the rotation group are written directly (no WHT)
    // TODO: support DV > 128 with multiple rotation groups if needed
}

// Template instantiations for TurboFlash Pass 1
// DK = head dim for K (q8_0), DV = head dim for V (turbo3)
// Common asymmetric config: DK == DV for most models
typedef decltype(kernel_turbo_flash_p1<128, 128>) turbo_flash_p1_t;

template [[host_name("kernel_turbo_flash_p1_dk64_dv64")]]   kernel turbo_flash_p1_t kernel_turbo_flash_p1<64,   64>;
template [[host_name("kernel_turbo_flash_p1_dk96_dv96")]]   kernel turbo_flash_p1_t kernel_turbo_flash_p1<96,   96>;
template [[host_name("kernel_turbo_flash_p1_dk128_dv128")]] kernel turbo_flash_p1_t kernel_turbo_flash_p1<128, 128>;

// Template instantiations for TurboFlash Pass 2
typedef decltype(kernel_turbo_flash_p2<128>) turbo_flash_p2_t;

template [[host_name("kernel_turbo_flash_p2_dv64")]]  kernel turbo_flash_p2_t kernel_turbo_flash_p2<64>;
template [[host_name("kernel_turbo_flash_p2_dv96")]]  kernel turbo_flash_p2_t kernel_turbo_flash_p2<96>;
template [[host_name("kernel_turbo_flash_p2_dv128")]] kernel turbo_flash_p2_t kernel_turbo_flash_p2<128>;

// ============================================================================
// End TurboFlash
// ============================================================================

