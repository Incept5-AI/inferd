// MLX's NAX bf16 GEMM (vendored, unmodified, third_party/mlx) for bf16 checkpoints: C[M, N] = A[M, K] @ W[N, K]^T.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h"

instantiate_kernel("mlx_gemm_bf16_nt_bm64_bn128_bk256", gemm, bfloat, 64, 128, 256, 2, 4, false, true, float)
instantiate_kernel("mlx_gemm_bf16_nt_bm128_bn128_bk512", gemm, bfloat, 128, 128, 512, 4, 4, false, true, float)
