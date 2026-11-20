// MLX's simdgroup-matrix quantized matmul (affine_qmm_t, vendored from third_party/mlx quantized.h, flip-aware via
// function constant 40), for GPUs without neural accelerators (M3/M4): the NAX qmm is wrong there and W4A8's matmul2d
// int8 path is emulated. affine, group 64, bfloat, transposed weights, unbatched, BM = BN = BK = 32, 4 simdgroups.
// Name: mlx_qmm_sg_b<bits>_al<aligned N>.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define QMMSG(bits, al) \
  instantiate_kernel("mlx_qmm_sg_b" #bits "_al" #al, affine_qmm_t, bfloat16_t, 64, bits, al, false)
QMMSG(4, true) QMMSG(4, false) QMMSG(8, true) QMMSG(8, false)

// Tile-shape variants (same kernel, BM x BK x BN template args; WM = WN = 2 simdgroups, so BM, BN are multiples of
// 16; BK <= the group size, 64).
// Name: mlx_qmm_sg_b<bits>_al<aligned N to BN>_<BM>x<BK>x<BN>.
#define QMMSGT(bits, al, bm, bk, bn) \
  instantiate_kernel("mlx_qmm_sg_b" #bits "_al" #al "_" #bm "x" #bk "x" #bn, affine_qmm_t, bfloat16_t, 64, bits, al, \
                     false, false, bm, bk, bn)
#define QMMSGT_AL(bm, bk, bn) QMMSGT(4, true, bm, bk, bn) QMMSGT(4, false, bm, bk, bn) \
  QMMSGT(8, true, bm, bk, bn) QMMSGT(8, false, bm, bk, bn)
// Kept: the shapes Qwen35::sg_tile picks (M3 Ultra sweep, 2026-11-01; BK = 32 and BM = 64 lost everywhere). With
// WM = WN = 2, BM * BK / 128 must divide BK into rows that divide BM, so BM is 16 or 32 at BK 64.
QMMSGT_AL(16, 64, 16) QMMSGT_AL(16, 64, 32) QMMSGT_AL(32, 64, 16) QMMSGT_AL(32, 64, 32)  // BM = 48 here drops x columns (MLX loader)

// MLX's affine_qmv_wide (what MLX itself runs for 2 <= M < ~13-32 rows on gen-15+ GPUs): each threadgroup streams
// NV input vectors and decodes each weight group once for all of them; k_lanes lanes reduce K per output row.
// grid (ceil(M / NV), ceil(N / (2 * 32 / KL))), threads (32, 2). Name: mlx_qmv_wide_b<bits>_nv<NV>_kl<KL>.
#define QMVW(bits, nv, kl) \
  instantiate_kernel("mlx_qmv_wide_b" #bits "_nv" #nv "_kl" #kl, affine_qmv_wide, bfloat16_t, 64, bits, nv, kl, false)
#define QMVW_ALL(bits, kl) QMVW(bits, 1, kl) QMVW(bits, 2, kl) QMVW(bits, 3, kl) QMVW(bits, 4, kl) \
  QMVW(bits, 5, kl) QMVW(bits, 6, kl) QMVW(bits, 7, kl) QMVW(bits, 8, kl)
QMVW_ALL(4, 8) QMVW_ALL(4, 16) QMVW_ALL(8, 8) QMVW_ALL(8, 16)

// qmm_t_impl with the simdgroup layout as template args (MLX fixes WM = WN = 2, so BM is a multiple of 16). WM = 1,
// WN = 4 allows BM in steps of 8: the GEMM is MMA-bound here (cost ~ 20 ms + 2.2 ms per tile row per 27B step on M3
// Ultra), so rounding M up to 8 instead of 16 or 32 is the gain. Same loaders and BlockMMA as MLX; flip-aware via the
// patched dequantize (function constant 40); buffer 8 = the weight row length, so it also serves K-slice views (the ANE
// split's down projection). Name: mlx_qmm_sgw_b<bits>_al<aligned N to BN>_<BM>x<BN>, BK = 64.
// The ordinary path rounds the projection to bf16 before adding it to the residual.
struct SgRoundedResidual {
  static float apply(float product, bfloat16_t residual) {
    return float(bfloat16_t(product)) + float(residual);
  }
};

template <typename T, const int group_size, const int bits, const bool aligned_N, const int BM, const int BK,
          const int BN, const int WM, const int WN, int residual_k = 0, int residual_n = 0>
[[kernel]] void qmm_t_wmwn(
    const device uint32_t* w [[buffer(0)]], const device T* scales [[buffer(1)]], const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]], device T* y [[buffer(4)]], const constant int& input_k [[buffer(5)]],
    const constant int& output_n [[buffer(6)]], const constant int& rows [[buffer(7)]],
    const constant int& weight_stride [[buffer(8)]],  // weight row length in elements: K, or the full K of a K-slice view
    uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int K = residual_k ? residual_k : input_k, N = residual_n ? residual_n : output_n;
  const int M = residual_k ? BM : rows, ldw = residual_k ? residual_k : weight_stride;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  // BlockLoader's default reads BM * BK / 128 values per thread, which leaves columns unloaded unless that divides BK
  // into a row count dividing BM (BM = 24/40/48 at BK 64 silently dropped 4-16 columns of x). 4 per thread tiles
  // any BM that is a multiple of 8.
  constexpr short nrd = BM * BK / (WM * WN * SIMD_SIZE);
  constexpr short nr = (BK % nrd == 0 && BM % ((WM * WN * SIMD_SIZE) / (BK / nrd)) == 0) ? nrd : 4;
  static_assert(BM % 8 == 0 && BK % 64 == 0, "qmm_t_wmwn: BM multiple of 8, BK 64");
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, nr>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = ldw * bytes_per_pack / pack_factor;
  const int K_g = ldw / group_size;
  // The fixed-shape residual dispatch covers all 48 rows with one tile vertically.
  // Making that explicit also lets the compiler omit the partial-row loader.
  const int y_row = residual_k ? 0 : int(tid.y) * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, ldw, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  const bool full_m = num_els == BM, full_n = aligned_N || num_outs == BN;
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full_m) loader_x.load_unsafe(); else loader_x.load_safe(short2(BK, num_els));
    if (full_n) loader_w.load_unsafe(); else loader_w.load_safe(short2(BK, num_outs));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  if constexpr (!residual_k) threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (residual_k != 0) mma_op.apply_epilogue(y, N, 1, SgRoundedResidual{});
  if (!full_m || !full_n) mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  else mma_op.store_result(y, N);
}

#define QMMSGW(bits, al, bm, bn) \
  template [[host_name("mlx_qmm_sgw_b" #bits "_al" #al "_" #bm "x" #bn)]] [[kernel]] void \
  qmm_t_wmwn<bfloat16_t, 64, bits, al, bm, 64, bn, 1, 4>( \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, const device bfloat16_t*, \
      device bfloat16_t*, const constant int&, const constant int&, const constant int&, const constant int&, uint3, \
      uint, uint);
#define QMMSGW_AL(bm, bn) QMMSGW(4, true, bm, bn) QMMSGW(4, false, bm, bn) QMMSGW(8, true, bm, bn) QMMSGW(8, false, bm, bn)
QMMSGW_AL(8, 32) QMMSGW_AL(16, 32) QMMSGW_AL(24, 32) QMMSGW_AL(32, 32) QMMSGW_AL(40, 32) QMMSGW_AL(48, 32)
QMMSGW_AL(56, 32)  // 64x32: 442 ms per 27B step at 64 rows against 406 for the 2x2 32x64x32 above (M3 Max)

template [[host_name("mlx_qmm_sgw_down48_residual")]] [[kernel]] void
qmm_t_wmwn<bfloat16_t, 64, 4, true, 48, 64, 32, 1, 4, 17408, 5120>(
    const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, const device bfloat16_t*,
    device bfloat16_t*, const constant int&, const constant int&, const constant int&, const constant int&, uint3,
    uint, uint);

template [[host_name("mlx_qmm_sgw_out48_residual")]] [[kernel]] void
qmm_t_wmwn<bfloat16_t, 64, 4, true, 48, 64, 32, 1, 4, 6144, 5120>(
    const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, const device bfloat16_t*,
    device bfloat16_t*, const constant int&, const constant int&, const constant int&, const constant int&, uint3,
    uint, uint);

// A separate unpack lets all row tiles share the same BF16 weights in a large prefill.
// dequantize uses the same affine BF16 arithmetic and flip flag as the packed simdgroup path.
kernel void prefill_unpack4(device const uchar* packed [[buffer(0)]], device const bfloat* scales [[buffer(1)]],
                            device const bfloat* biases [[buffer(2)]], device bfloat* weights [[buffer(3)]],
                            uint pack [[thread_position_in_grid]]) {
  dequantize<bfloat, 8, 4>(packed + 4 * pack, scales[pack / 8], biases[pack / 8], weights + 8 * pack);
}

kernel void prefill_dense_gdn(device const bfloat* input [[buffer(0)]], device const bfloat* weights [[buffer(1)]],
                              device bfloat* output [[buffer(2)]], constant int& M [[buffer(3)]],
                              constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
                              uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64, BN = 64, BK = 32, stride = BK + 8;
  threadgroup bfloat xs[BM * stride], ws[BN * stride];
  using XLoader = mlx::steel::BlockLoader<bfloat, BM, BK, stride, 1, 128, 1, 4>;
  using WLoader = mlx::steel::BlockLoader<bfloat, BN, BK, stride, 1, 128, 1, 4>;
  using MMA = mlx::steel::BlockMMA<bfloat, bfloat, BM, BN, BK, 2, 2, false, true, stride, stride>;
  const int row = group.y * BM, col = group.x * BN;
  const short rows = min(BM, M - row), cols = min(BN, N - col);
  XLoader x(input + size_t(row) * K, K, xs, simd, lane);
  WLoader w(weights + size_t(col) * K, K, ws, simd, lane);
  MMA mma(simd, lane);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rows == BM) x.load_unsafe(); else x.load_safe(short2(BK, rows));
    if (cols == BN) w.load_unsafe(); else w.load_safe(short2(BK, cols));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(xs, ws);
    x.next();
    w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  output += size_t(row) * N + col;
  if (rows == BM && cols == BN) mma.store_result(output, N);
  else mma.store_result_safe(output, N, short2(cols, rows));
}

// Two workgroups share one 48-row down-projection output tile. Each computes half of K
// into the existing FP32 split-K workspace. The change in summation order can affect greedy output.
kernel void sg_down48_split2(device const uint8_t* packed [[buffer(0)]], device const bfloat* scales [[buffer(1)]],
                             device const bfloat* biases [[buffer(2)]], device const bfloat* input [[buffer(3)]],
                             device float* partial [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int M = 48, N = 5120, K = 17408, BN = 32, BK = 64, stride = BK + 8, slice = K / 2;
  threadgroup bfloat xs[M * stride], ws[BN * stride];
  using XLoader = mlx::steel::BlockLoader<bfloat, M, BK, stride, 1, 128, 1, 4>;
  using WLoader = QuantizedBlockLoader<bfloat, BN, BK, stride, 1, 128, 64, 4>;
  using MMA = mlx::steel::BlockMMA<bfloat, float, M, BN, BK, 2, 2, false, true, stride, stride>;
  const int col = group.x * BN, start = group.z * slice;
  const size_t offset = size_t(col) * K + start;
  XLoader x(input + start, K, xs, sg, lane);
  WLoader w(packed + offset / 2, scales + offset / 64, biases + offset / 64, K, ws, sg, lane);
  MMA mma(sg, lane);
  for (int k = 0; k < slice; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x.load_unsafe();
    w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(xs, ws);
    x.next();
    w.next();
  }
  mma.store_result(partial + group.z * M * N + col, N);
}

kernel void sg_down48_split2_residual(device const float* partial [[buffer(0)]], device bfloat* residual [[buffer(1)]],
                                      uint i [[thread_position_in_grid]]) {
  constexpr int elements = 48 * 5120;
  const float sum = partial[i] + partial[elements + i];
  residual[i] = bfloat(SgRoundedResidual::apply(sum, residual[i]));
}
