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
template <typename T, const int group_size, const int bits, const bool aligned_N, const int BM, const int BK,
          const int BN, const int WM, const int WN>
[[kernel]] void qmm_t_wmwn(
    const device uint32_t* w [[buffer(0)]], const device T* scales [[buffer(1)]], const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]], device T* y [[buffer(4)]], const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]], const constant int& M [[buffer(7)]],
    const constant int& ldw [[buffer(8)]],  // weight row length in elements: K, or the full K of a K-slice view
    uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
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
  const int y_row = tid.y * BM;
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
