// MLX's NAX gathered quantized matmul over expert-sorted rows (vendored from third_party/mlx; see its README for our patches):
// affine, group 64, bfloat, transposed weights [E, N, K]. Name: mlx_gqmm_b<bits>_bm<BM>.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define GQMM(bits, bm, wm, wn) \
  instantiate_kernel("mlx_gqmm_b" #bits "_bm" #bm, affine_gather_qmm_rhs_nax, bfloat16_t, 64, bits, bm, 64, 64, wm, wn, true)
GQMM(4, 16, 1, 2) GQMM(4, 64, 2, 2) GQMM(8, 16, 1, 2) GQMM(8, 64, 2, 2)
