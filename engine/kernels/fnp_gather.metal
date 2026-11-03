// Flash-Next prefill: more row-tile shapes of MLX's NAX gathered quantized matmul (mlx_gather.metal's template,
// vendored unmodified in third_party/mlx). Name: mlx_gqmm_b<bits>_bm<BM>[_w<WM>].
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define GQMMW(bits, bm, wm, wn, sfx) \
  instantiate_kernel("mlx_gqmm_b" #bits "_bm" #bm sfx, affine_gather_qmm_rhs_nax, bfloat16_t, 64, bits, bm, 64, 64, wm, wn, true)
GQMMW(4, 32, 1, 2, "") GQMMW(4, 32, 2, 2, "_w2") GQMMW(4, 128, 4, 2, "") GQMMW(4, 64, 2, 4, "_n4")
