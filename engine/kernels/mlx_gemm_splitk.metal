// MLX's NAX split-K bf16 GEMM (vendored, unmodified) for bf16 checkpoints at small M, where the output has too few
// tiles to fill the GPU: float partials [S, M, N], summed by splitk_reduce (qmm_splitk.metal).
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.h"

instantiate_kernel("mlx_gemm_splitk_bf16_nt_bm64_bn64_bk256", gemm_splitk_nax, bfloat, 64, 64, 256, 2, 2, false, true, float)
