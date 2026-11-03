// Split-K quantized matmul for narrow outputs (N = hidden size), where one K-pass per output tile leaves
// most of the GPU idle. Derived from MLX's qmm_t_nax_tgp_impl (third_party/mlx, MIT): same loader and NAX
// tiles, but each threadgroup covers K-slice tid.z and writes float partials [S, M, N]; splitk_reduce sums them.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

using namespace mlx::steel;

struct SplitArgs { int M, N, K, k_len; };

template <typename T, int group_size, int bits, int BM, int BK = 64, int BN = 64, int WM = 2, int WN = 2>
[[kernel]] void qmm_splitk(const device uint32_t* w [[buffer(0)]],
                           const device T* scales [[buffer(1)]],
                           const device T* biases [[buffer(2)]],
                           const device T* x [[buffer(3)]],
                           device float* y [[buffer(4)]],
                           constant SplitArgs& p [[buffer(5)]],
                           uint3 tid [[threadgroup_position_in_grid]],
                           uint simd_gid [[simdgroup_index_in_threadgroup]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Ws[BN * BK_padded];
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

  const int M = p.M, N = p.N, K = p.K;
  const int k0 = tid.z * p.k_len;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K) + k0;
  wl += y_col * K_w + k0 * bytes_per_pack / pack_factor;
  scales += y_col * K_g + k0 / group_size;
  biases += y_col * K_g + k0 / group_size;
  y += (size_t)tid.z * M * N + y_row * static_cast<int64_t>(N) + y_col;

  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);
  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);
  const short sgp_sn = min(int(SN), N - (y_col + tn));
  const short tgp_bn = min(BN, int(N - y_col));
  const bool is_unaligned_bn = tgp_bn != BN;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(!is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < p.k_len; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) loader_w.load_unsafe();
        else loader_w.load_safe(short2(BK, tgp_bn));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;
          volatile int compiler_barrier;
          if constexpr (kAlignedM.value) Atile.load(x + kk1, K);
          else Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
          tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
          (void)compiler_barrier;
        }
        x += BK;
        loader_w.next();
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (kAlignedM.value && kAlignedN.value) Dtile.store(y + tm * N + tn, N);
      else Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    });
  });
}

#define SPLITK(bits, bm) \
  instantiate_kernel("qmm_splitk_b" #bits "_bm" #bm, qmm_splitk, bfloat16_t, 64, bits, bm)
SPLITK(4, 32) SPLITK(4, 64) SPLITK(8, 32) SPLITK(8, 64) SPLITK(5, 32) SPLITK(5, 64) SPLITK(6, 32) SPLITK(6, 64)

// out[m, n] (+)= sum_s part[s, m, n]. grid (N, M).
struct ReduceArgs { int M, N, S, add; };
[[kernel]] void splitk_reduce(const device float* part [[buffer(0)]],
                              device bfloat16_t* out [[buffer(1)]],
                              constant ReduceArgs& p [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if ((int)gid.x >= p.N) return;
  size_t o = (size_t)gid.y * p.N + gid.x, mn = (size_t)p.M * p.N;
  float s = 0;
  for (int i = 0; i < p.S; i++) s += part[i * mn + o];
  out[o] = bfloat16_t(p.add ? s + float(out[o]) : s);
}
