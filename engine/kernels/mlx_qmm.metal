// MLX's NAX quantized matmul (vendored from third_party/mlx; see its README for our patches), instantiated for this engine:
// affine, group 64, bfloat, transposed weights, unbatched. Name: mlx_qmm_b<bits>_bm<BM>_al<aligned N>.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define QMM(bits, bm, al) \
  instantiate_kernel("mlx_qmm_b" #bits "_bm" #bm "_al" #al, affine_qmm_t_nax, bfloat16_t, 64, bits, al, false, bm, 64, 64, 2, 2)
QMM(4, 32, true) QMM(4, 32, false) QMM(4, 64, true) QMM(4, 64, false)
QMM(8, 32, true) QMM(8, 32, false) QMM(8, 64, true) QMM(8, 64, false)
QMM(5, 32, true) QMM(5, 32, false) QMM(5, 64, true) QMM(5, 64, false)
QMM(6, 32, true) QMM(6, 32, false) QMM(6, 64, true) QMM(6, 64, false)
