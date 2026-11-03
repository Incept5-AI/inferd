// MLX's NAX flash attention (vendored, third_party/mlx; patched with a sliding window) for prefill and multi-row blocks:
// bfloat, head dim 256 (the d-split variant), causal with queries at the end of the key range.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

instantiate_kernel("mlx_attn_bf16_d256", attention_nax_dsplit, bfloat16_t, 64, 32, 256, 4, 2, bfloat16_t, float)
// Head dim 128 (Muse Glimmer): the plain NAX kernel, 4 simdgroups over 64 query rows.
instantiate_kernel("mlx_attn_bf16_d128", attention_nax, bfloat16_t, 64, 32, 128, 128, 4, 1, bfloat16_t, float)
// Head dim 512 (Gemma-4 global layers): the d-split kernel, 4 simdgroups over the head dim.
instantiate_kernel("mlx_attn_bf16_d512", attention_nax_dsplit, bfloat16_t, 64, 32, 512, 4, 4, bfloat16_t, float)
