// QA8 (engine-a8, 2026-11-04): int8-activation prefill GEMM that reads the checkpoint's PACKED 4-bit weights directly.
//
// Adapted from oMLX's `oq_a8_qmm_t_nax_v8` (omlx/custom_kernels/quant_matmul/csrc/oq_a8_nax.metal and the Stage-A
// quantizer in oq_a8.metal / fast.py `oq_a8_stage_a_v8`, github.com/jundot/omlx at v0.7.4.dev1, commit 4213c32,
// Copyright 2025 oMLX contributors, Apache License 2.0; licence and the list of changes in third_party/omlx/, credited
// in NOTICE). This file is a modified adaptation (Incept5, 2026), not a verbatim copy. What is theirs: the
// register-fragment design (16x32x16 int8 matmul2d per simdgroup, weights decoded from the packed nibbles straight into the right-operand cooperative tensor, int32 accumulation over one 64-group in four
// 16-deep steps, then the affine correction in fp32), the per-lane K schedule ("step t of a lane's 16-code run takes
// nibbles 8(t>>1) + 2j + (t&1)", one mask or shift per word) and the matching permutation of the activations.
// What differs here (inferd's layouts and quality rules):
//   * activations: one int8 scale per row per 64-group (oMLX's ACT_MODE 1). Per-row scales (oMLX's Q4 default) cost
//     KL 0.049 / top-1 89% on the 27B (NOTES 2026-10-30), which inferd does not accept.
//   * weights: MLX's row-major [N][K/8] words, possibly flipped in place for W4A8 (q ^ 8): one XOR per word undoes it.
//   * scales/biases: read from the checkpoint's N-major [N][K/64] layout, 4 groups at a time, into a double-buffered
//     threadgroup stage (group-major), instead of oMLX's transposed metadata copy. Strided per-lane loads cost ~40%.
//   * epilogues: bf16 store, bf16 residual add, float store (inferd's GEMM modes); grid order chosen by the host.
//
//   y[m, n] = sum_g sx[m, g] * (S[n, g] * sum_k q[n, k] a[m, k] + B[n, g] * sum_k a[m, k])     (k in group g)
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
using namespace metal;
using namespace mlx::steel;

constant constexpr int QG = 64;  // affine group (MLX group_size 64)

// Activations: one simdgroup per (row, group), 2 values per lane. Writes int8 in the GEMM's permuted K order (within
// each 16-run, slot 4t + j holds k = 8(t>>1) + 2j + (t&1)) and XS[g][m] = (sx, sx * sum q) group-major, so the GEMM's
// per-group metadata loads are contiguous across rows. grid threads (K/64 * 32, M), threadgroups of 256.
struct Qa8QArgs { int M, K, lda; };
kernel void qa8_quant(device const bfloat* A [[buffer(0)]], device char* U [[buffer(1)]],
                      device float2* XS [[buffer(2)]], constant Qa8QArgs& p [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const int m = gid.y, g = gid.x / 32, kg = p.K / QG;
  if (m >= p.M || g >= kg) return;
  const int k0 = int(lane) * 2;  // within the group
  device const bfloat* a = A + (size_t)m * p.lda + g * QG + k0;
  const float2 v = float2(a[0], a[1]);
  const float amax = simd_max(max(fabs(v.x), fabs(v.y)));
  const float inv = amax > 0 ? 127.f / amax : 0.f;
  const float q0 = clamp(rint(v.x * inv), -127.f, 127.f), q1 = clamp(rint(v.y * inv), -127.f, 127.f);
  // k0 even: r = k0 & 15, t = 2 (r >> 3), j = (r & 7) >> 1; k0 + 1: t + 1
  const int r = k0 & 15, c = k0 >> 4;
  const int slot = 16 * c + 8 * (r >> 3) + ((r & 7) >> 1);
  device char* u = U + (size_t)m * p.K + g * QG;
  u[slot] = char(q0);
  u[slot + 4] = char(q1);
  const float qs = simd_sum(q0 + q1);
  if (lane == 0) XS[(size_t)g * p.M + m] = float2(amax / 127.f, amax / 127.f * qs);
}

// C[M, N] (op)= A8 . dequant(W4)^T. WM x WN simdgroups, each 32 rows (two 16-row fragments) x 32 columns.
// MODE 0: C = bf16, 1: C += bf16 (residual), 2: C = float. ROWFAST: threadgroup x walks the row tiles (consecutive
// threadgroups share a weight tile), else x walks the column tiles (oMLX's order). Requires N % 32 == 0, K % 256 == 0.
struct Qa8GArgs { int M, N, K; uint xr; int tiles_x; };
template <int WM, int WN, int MODE, bool ROWFAST>
[[kernel, max_total_threads_per_threadgroup(32 * WM * WN)]] void qa8_gemm(
    device const char* qa [[buffer(0)]], device const uint* w [[buffer(1)]], device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]], device const float2* xs [[buffer(4)]], device void* Cv [[buffer(5)]],
    constant Qa8GArgs& p [[buffer(6)]], uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]], uint lid [[thread_index_in_threadgroup]]) {
  constexpr int TM = 2, FM = 16, FN = 32, FK = 16, EPF = 8, DE = 2 * EPF, SPG = QG / FK;
  constexpr int BM = TM * FM * WM, BN = FN * WN, WORDS = QG * 4 / 32;  // 8 words per group
  constexpr int GB = 4, NT = 32 * WM * WN;  // groups per metadata block (K % 256 == 0), threads
  const int M = p.M, N = p.N, K = p.K, groups = K / QG;
  const uint xr = p.xr;
  const int sg_m = int(simd_gid) % WM, sg_n = int(simd_gid) / WM;
  const int tx = ROWFAST ? int(tid.x) % p.tiles_x : int(tid.x), ty = ROWFAST ? int(tid.x) / p.tiles_x : int(tid.y);
  // ROWFAST: x = row tile (fast), y = column tile (via the flattened index); else x = column tile, y = row tile
  const int rt = ROWFAST ? tx : ty, ct = ROWFAST ? ty : tx;
  const int row_base = rt * BM + sg_m * (TM * FM);
  // N % 32 == 0: a partial column tile's spare simdgroups compute a valid tile (the last one) and store nothing; they
  // must stay for the threadgroup barriers.
  const bool live = ct * BN + sg_n * FN < N;
  const int col_base = live ? ct * BN + sg_n * FN : N - FN;
  const short2 coord = BaseNAXFrag::get_coord();

  // The threadgroup's scales and biases for GB groups, group-major [GB][BN] (one 8-byte load per column per block from
  // the N-major checkpoint layout), double-buffered. 8 lanes of every simdgroup (and WM simdgroups) share each column.
  threadgroup bfloat tS[2][GB][BN], tB[2][GB][BN];
  const int tcol0 = ct * BN;
  auto stage = [&](int buf, int gb) {
    for (int i = int(lid); i < 2 * BN; i += NT) {
      const int c = i % BN, which = i / BN, n = min(tcol0 + c, N - 1);
      const vec<bfloat, 4> v = *reinterpret_cast<device const vec<bfloat, 4>*>((which ? biases : scales) + (size_t)n * groups + gb * GB);
      threadgroup bfloat(*t)[GB][BN] = which ? tB : tS;
      t[buf][0][c] = v.x, t[buf][1][c] = v.y, t[buf][2][c] = v.z, t[buf][3][c] = v.w;
    }
  };

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(FM, FN, FK, false, true, false,
                                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(FM, FN, FK, false, true, false,
                                                                 mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  const int m_base = row_base + int(coord.y);  // destination rows: m_base + 16 i + 8 r
  const int tc = col_base - tcol0 + int(coord.x);  // this lane's first column in the staged block
  float Cf[TM][DE];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i)
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < DE; ++e) Cf[i][e] = 0.0f;

  // Right operand rows of this lane: n_lane + 8 (q & 1) + 16 (q >> 1)
  const int n_lane = col_base + int(coord.y);
  device const uint* wbase = w + (size_t)n_lane * groups * WORDS;
  const int w_row = groups * WORDS;
  const int cx = int(coord.x) >> 2;  // which 16-code run of the group this lane owns

  stage(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int nblk = groups / GB;
  for (int gb = 0; gb < nblk; ++gb) {
    const int buf = gb & 1;
    if (gb + 1 < nblk) stage(buf ^ 1, gb + 1);
    for (int gg = 0; gg < GB; ++gg) {
      const int g = gb * GB + gg;
      uint2 wg[4];
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        device const uint* wr = wbase + (size_t)((q & 1) * 8 + (q >> 1) * 16) * w_row + g * WORDS;
        wg[q] = reinterpret_cast<device const uint2*>(wr)[cx] ^ uint2(xr);
      }
      STEEL_PRAGMA_UNROLL
      for (int t = 0; t < SPG; ++t) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < 4; ++q) {
          const int base = (q >> 1) * EPF + (q & 1) * 4;
          const uint word = wg[q][t >> 1];
          const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
          ct_b[base + 0] = quad.x;
          ct_b[base + 1] = quad.y;
          ct_b[base + 2] = quad.z;
          ct_b[base + 3] = quad.w;
        }
        STEEL_PRAGMA_UNROLL
        for (int hf = 0; hf < 2; ++hf) {
          STEEL_PRAGMA_UNROLL
          for (int r = 0; r < 2; ++r) {
            const int m = min(m_base + hf * FM + r * 8, M - 1);
            const char4 quad = as_type<char4>(*reinterpret_cast<device const uint*>(
                qa + (size_t)m * K + (size_t)g * QG + cx * 16 + t * 4));
            ct_a[r * 4 + 0] = quad.x;
            ct_a[r * 4 + 1] = quad.y;
            ct_a[r * 4 + 2] = quad.z;
            ct_a[r * 4 + 3] = quad.w;
          }
          if (t == 0) {
            if (hf == 0) op_set.run(ct_a, ct_b, acc0);
            else op_set.run(ct_a, ct_b, acc1);
          } else {
            if (hf == 0) op.run(ct_a, ct_b, acc0);
            else op.run(ct_a, ct_b, acc1);
          }
        }
      }
      // Affine correction for group g: S, B of this lane's 8 columns (tc + 16 h + j) from the staged block
      vec<bfloat, 4> sv[2], bv[2];
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        sv[h] = *reinterpret_cast<threadgroup const vec<bfloat, 4>*>(&tS[buf][gg][tc + 16 * h]);
        bv[h] = *reinterpret_cast<threadgroup const vec<bfloat, 4>*>(&tB[buf][gg][tc + 16 * h]);
      }
      device const float2* xrow = xs + (size_t)g * M;
      float2 x_g[TM][2];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i)
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) x_g[i][r] = xrow[min(m_base + i * FM + r * 8, M - 1)];
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < DE; ++e) {
        const int r = (e & 7) >> 2;
        const float swc = float(sv[e >> 3][e & 3]), bwc = float(bv[e >> 3][e & 3]);
        Cf[0][e] = metal::fma(swc * x_g[0][r].x, float(acc0[e]), metal::fma(bwc, x_g[0][r].y, Cf[0][e]));
        Cf[1][e] = metal::fma(swc * x_g[1][r].x, float(acc1[e]), metal::fma(bwc, x_g[1][r].y, Cf[1][e]));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!live) return;

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i)
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < DE; ++e) {
      const int ee = e & 7, r = ee >> 2;
      const int m = m_base + i * FM + r * 8;
      if (m < M) {
        const int n = col_base + (e >> 3) * FM + int(coord.x) + (ee & 3);
        const size_t o = (size_t)m * N + n;
        const float v = Cf[i][e];
        if (MODE == 2) ((device float*)Cv)[o] = v;
        else if (MODE == 1) ((device bfloat*)Cv)[o] = bfloat(v + float(((device bfloat*)Cv)[o]));
        else ((device bfloat*)Cv)[o] = bfloat(v);
      }
    }
}

#define QA8G(WM, WN, MODE, RF)                                                                                       \
  template [[host_name("qa8_" #WM "x" #WN "_m" #MODE "_rf" #RF)]] [[kernel]] void qa8_gemm<WM, WN, MODE, RF>(           \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*,       \
      device void*, constant Qa8GArgs&, uint3, uint, uint);
#define QA8T(WM, WN) QA8G(WM, WN, 0, 0) QA8G(WM, WN, 1, 0) QA8G(WM, WN, 2, 0) QA8G(WM, WN, 0, 1) QA8G(WM, WN, 1, 1) QA8G(WM, WN, 2, 1)
// (BM x BN): 1x2 32x64 (oMLX's Q4 default, 806), 2x2 64x64, 4x2 128x64, 2x4 64x128, 4x4 128x128, 1x4 32x128, 8x2 256x64
QA8T(1, 2) QA8T(2, 2) QA8T(4, 2) QA8T(2, 4) QA8T(4, 4) QA8T(1, 4) QA8T(8, 2)

// QA8G (2026-11-05): qa8_gemm over expert-sorted rows for the MoE experts (Qwen3.6-35B-A3B prefill). Rows [0, M) are
// sorted by expert (moe_sort: offsets[e] = expert e's first row); each threadgroup takes one row tile of one expert
// (MLX's schedule_row_tile: tiles = ceil(rows_e / BM) per expert, in expert order) and one column tile of that
// expert's [N][K] slice of the stacked weights. GA: the int8 activations are the T unsorted token rows (quantized
// once) and sorted row m reads token perm[m] / div (no gathered bf16 copy); else row m reads activation row m.
// XS is group-major with stride MQ (the number of quantized rows). Output: bf16 [M, N] in sorted order.
struct Qa8GGArgs { int M, N, K; uint xr; int E, div, MQ; };
template <int WM, int WN, bool GA>
[[kernel, max_total_threads_per_threadgroup(32 * WM * WN)]] void qa8g_gemm(
    device const char* qa [[buffer(0)]], device const uint* w [[buffer(1)]], device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]], device const float2* xs [[buffer(4)]], device bfloat* C [[buffer(5)]],
    constant Qa8GGArgs& p [[buffer(6)]], device const int* offsets [[buffer(7)]], device const int* perm [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int TM = 2, FM = 16, FN = 32, FK = 16, EPF = 8, DE = 2 * EPF, SPG = QG / FK;
  constexpr int BM = TM * FM * WM, BN = FN * WN, WORDS = QG * 4 / 32;
  constexpr int GB = 4, NT = 32 * WM * WN;
  const int M = p.M, N = p.N, K = p.K, groups = K / QG;
  const uint xr = p.xr;

  // Which expert and rows this row tile covers (every simdgroup computes the same answer).
  int row0 = 0, expert = -1, nrows = 0;
  {
    int before = 0;
    const int tile = int(tid.y);
    for (int e0 = 0; e0 < p.E; e0 += 32) {
      const int e = e0 + int(lane);
      int s = M, en = M;
      if (e < p.E) { s = offsets[e]; en = e + 1 < p.E ? offsets[e + 1] : M; }
      const int n = (en - s + BM - 1) / BM;
      const int through = before + simd_prefix_inclusive_sum(n);
      const int owner = simd_sum(int(through <= tile));
      if (owner < 32) {
        expert = e0 + owner;
        row0 = simd_shuffle(s, ushort(owner)) + (tile - simd_shuffle(through - n, ushort(owner))) * BM;
        nrows = min(BM, simd_shuffle(en, ushort(owner)) - row0);
        break;
      }
      before = simd_shuffle(through, ushort(31));
    }
  }
  if (expert < 0 || nrows <= 0) return;
  const int row_end = row0 + nrows;
  w += (size_t)expert * N * groups * WORDS;
  scales += (size_t)expert * N * groups;
  biases += (size_t)expert * N * groups;

  const int sg_m = int(simd_gid) % WM, sg_n = int(simd_gid) / WM;
  const int ct = int(tid.x);
  const int row_base = row0 + sg_m * (TM * FM);
  const bool live = ct * BN + sg_n * FN < N;
  const int col_base = live ? ct * BN + sg_n * FN : N - FN;
  const short2 coord = BaseNAXFrag::get_coord();

  threadgroup bfloat tS[2][GB][BN], tB[2][GB][BN];
  const int tcol0 = ct * BN;
  auto stage = [&](int buf, int gb) {
    for (int i = int(lid); i < 2 * BN; i += NT) {
      const int c = i % BN, which = i / BN, n = min(tcol0 + c, N - 1);
      const vec<bfloat, 4> v = *reinterpret_cast<device const vec<bfloat, 4>*>((which ? biases : scales) + (size_t)n * groups + gb * GB);
      threadgroup bfloat(*t)[GB][BN] = which ? tB : tS;
      t[buf][0][c] = v.x, t[buf][1][c] = v.y, t[buf][2][c] = v.z, t[buf][3][c] = v.w;
    }
  };

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(FM, FN, FK, false, true, false,
                                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(FM, FN, FK, false, true, false,
                                                                 mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  const int m_base = row_base + int(coord.y);  // destination rows: m_base + 16 i + 8 r
  // source activation rows of this lane's four destination rows (clamped into the tile)
  int src[TM][2];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i)
    STEEL_PRAGMA_UNROLL
    for (int r = 0; r < 2; ++r) {
      const int m = min(m_base + i * FM + r * 8, row_end - 1);
      src[i][r] = GA ? perm[m] / p.div : m;
    }
  const int tc = col_base - tcol0 + int(coord.x);
  float Cf[TM][DE];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i)
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < DE; ++e) Cf[i][e] = 0.0f;

  const int n_lane = col_base + int(coord.y);
  device const uint* wbase = w + (size_t)n_lane * groups * WORDS;
  const int w_row = groups * WORDS;
  const int cx = int(coord.x) >> 2;

  stage(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int nblk = groups / GB;
  for (int gb = 0; gb < nblk; ++gb) {
    const int buf = gb & 1;
    if (gb + 1 < nblk) stage(buf ^ 1, gb + 1);
    for (int gg = 0; gg < GB; ++gg) {
      const int g = gb * GB + gg;
      uint2 wg[4];
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        device const uint* wr = wbase + (size_t)((q & 1) * 8 + (q >> 1) * 16) * w_row + g * WORDS;
        wg[q] = reinterpret_cast<device const uint2*>(wr)[cx] ^ uint2(xr);
      }
      STEEL_PRAGMA_UNROLL
      for (int t = 0; t < SPG; ++t) {
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < 4; ++q) {
          const int base = (q >> 1) * EPF + (q & 1) * 4;
          const uint word = wg[q][t >> 1];
          const char4 quad = as_type<char4>(((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
          ct_b[base + 0] = quad.x;
          ct_b[base + 1] = quad.y;
          ct_b[base + 2] = quad.z;
          ct_b[base + 3] = quad.w;
        }
        STEEL_PRAGMA_UNROLL
        for (int hf = 0; hf < 2; ++hf) {
          STEEL_PRAGMA_UNROLL
          for (int r = 0; r < 2; ++r) {
            const char4 quad = as_type<char4>(*reinterpret_cast<device const uint*>(
                qa + (size_t)src[hf][r] * K + (size_t)g * QG + cx * 16 + t * 4));
            ct_a[r * 4 + 0] = quad.x;
            ct_a[r * 4 + 1] = quad.y;
            ct_a[r * 4 + 2] = quad.z;
            ct_a[r * 4 + 3] = quad.w;
          }
          if (t == 0) {
            if (hf == 0) op_set.run(ct_a, ct_b, acc0);
            else op_set.run(ct_a, ct_b, acc1);
          } else {
            if (hf == 0) op.run(ct_a, ct_b, acc0);
            else op.run(ct_a, ct_b, acc1);
          }
        }
      }
      vec<bfloat, 4> sv[2], bv[2];
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        sv[h] = *reinterpret_cast<threadgroup const vec<bfloat, 4>*>(&tS[buf][gg][tc + 16 * h]);
        bv[h] = *reinterpret_cast<threadgroup const vec<bfloat, 4>*>(&tB[buf][gg][tc + 16 * h]);
      }
      device const float2* xrow = xs + (size_t)g * p.MQ;
      float2 x_g[TM][2];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i)
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) x_g[i][r] = xrow[src[i][r]];
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < DE; ++e) {
        const int r = (e & 7) >> 2;
        const float swc = float(sv[e >> 3][e & 3]), bwc = float(bv[e >> 3][e & 3]);
        Cf[0][e] = metal::fma(swc * x_g[0][r].x, float(acc0[e]), metal::fma(bwc, x_g[0][r].y, Cf[0][e]));
        Cf[1][e] = metal::fma(swc * x_g[1][r].x, float(acc1[e]), metal::fma(bwc, x_g[1][r].y, Cf[1][e]));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!live) return;

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i)
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < DE; ++e) {
      const int ee = e & 7, r = ee >> 2;
      const int m = m_base + i * FM + r * 8;
      if (m < row_end) {
        const int n = col_base + (e >> 3) * FM + int(coord.x) + (ee & 3);
        C[(size_t)m * N + n] = bfloat(Cf[i][e]);
      }
    }
}

#define QA8GG(WM, WN, GA)                                                                                            \
  template [[host_name("qa8g_" #WM "x" #WN "_ga" #GA)]] [[kernel]] void qa8g_gemm<WM, WN, GA>(                        \
      device const char*, device const uint*, device const bfloat*, device const bfloat*, device const float2*,       \
      device bfloat*, constant Qa8GGArgs&, device const int*, device const int*, uint3, uint, uint, uint);
QA8GG(1, 2, 0) QA8GG(1, 2, 1) QA8GG(1, 4, 0) QA8GG(1, 4, 1) QA8GG(2, 2, 0) QA8GG(2, 2, 1) QA8GG(2, 4, 0) QA8GG(2, 4, 1)
