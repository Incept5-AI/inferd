// Gathered quantized matmul over expert-sorted rows on GPUs WITHOUT neural accelerators (M1-M4, !Gpu::nax).
// MLX's NAX gathered qmm (kernels/mlx_gather.metal, fnp_gather.metal; fp_gather_qmm_rhs_nax in mlx_fpq_nax.metal)
// returns garbage there: relative RMS error 1.4 on the M4 Pro at the 35B-A3B's shapes (tools/gqmm_test.mm, 21 Nov
// 2026; results/m4-35b-moe-fix-2026-11-21). Every sorted-MoE caller branches here when !g.nax; the NAX path is
// unchanged. The kernel is MLX's simdgroup affine_gather_qmm_rhs with MLX's own non-NAX tiles (BM 16, BN 32, BK 32,
// 1 x 2 simdgroups = 64 threads; mlx/backend/metal/quantized.cpp gather_qmm_rhs), what mlx_lm runs on these GPUs.
// Same buffers as the NAX kernel: 0 A [R, K] bf16 sorted rows, 1-3 weights [E, N, K] (affine, group 64, `bits` 4/8),
// 4 offs [E] (moe_sort), 5 C [R, N] bf16.
#pragma once
#include <algorithm>
#include <cstdio>
#include "../gpu.h"

inline void gqmm_sg(Gpu& g, Enc& e, const BufRef& A, const BufRef& wq, const BufRef& s, const BufRef& b,
                    const BufRef& offs, const BufRef& C, int R, int N, int K, int E, int bits, bool flipped = false) {
  char nm[24];
  snprintf(nm, sizeof nm, "mlx_gqmm_sg_b%d", bits);
  e.k(g.pso_fc(nm, {{40, flipped}, {201, N % 32 == 0}, {202, K % 32 == 0}}))
      .buf(0, A).buf(1, wq).buf(2, s).buf(3, b).buf(4, offs).buf(5, C).args(6, R).args(7, N).args(8, K).args(9, E);
  e.groups(S3((N + 31) / 32, (R + 15) / 16 + std::min(E, R)), S3(64));
}
