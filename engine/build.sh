#!/bin/zsh
# Build the kernels and tools on the machine's own macOS. Needs Xcode's Metal toolchain (the kernels are Metal 4).
# Do NOT lower the kernels' target (-mmacosx-version-min / MACOSX_DEPLOYMENT_TARGET): a macOS 26 target compiled MLX's
# NAX matmuls into kernels that give WRONG logits on macOS 27 at >= 256 rows (2026-11-03, reverted the same day).
set -e
cd "$(dirname $0)"
mkdir -p build
xcrun metal -std=metal4.0 -O3 -c kernels/core.metal -o build/core.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/mlx_qmm.metal -o build/mlx_qmm.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/qmm_splitk.metal -o build/qmm_splitk.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/mlx_qmm_sg.metal -o build/mlx_qmm_sg.air
xcrun metal -std=metal4.0 -O3 -c kernels/spec.metal -o build/spec.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/mlx_attn.metal -o build/mlx_attn.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/attn_paged.metal -o build/attn_paged.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/mlx_gemm.metal -o build/mlx_gemm.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/mlx_gemm_splitk.metal -o build/mlx_gemm_splitk.air
xcrun metal -std=metal4.0 -O3 -c kernels/w4a8.metal -o build/w4a8.air
xcrun metal -std=metal4.0 -O3 -c kernels/lane.metal -o build/lane.air
xcrun metal -std=metal4.0 -O3 -c kernels/ane.metal -o build/ane.air
xcrun metal -std=metal4.0 -O3 -c kernels/attn_mpp.metal -o build/attn_mpp.air
xcrun metal -std=metal4.0 -O3 -c kernels/qmst.metal -o build/qmst.air
xcrun metal -std=metal4.0 -O3 -c kernels/moe.metal -o build/moe.air
xcrun metal -std=metal4.0 -O3 -c kernels/qwen4.metal -o build/qwen4.air
xcrun metal -std=metal4.0 -O3 -c kernels/qmvb.metal -o build/qmvb.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/mlx_gather.metal -o build/mlx_gather.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/fnp_gather.metal -o build/fnp_gather.air
xcrun metal -std=metal4.0 -O3 -c kernels/tree.metal -o build/tree.air
xcrun metal -std=metal4.0 -O3 -c kernels/pf8.metal -o build/pf8.air
xcrun metal -std=metal4.0 -O3 -I third_party/mlx -c kernels/qa8.metal -o build/qa8.air
xcrun metallib build/tree.air build/qmvb.air build/qwen4.air build/mlx_gather.air build/moe.air build/w4a8.air build/lane.air build/ane.air build/attn_mpp.air build/qmst.air build/core.air build/mlx_qmm.air build/mlx_qmm_sg.air build/qmm_splitk.air build/spec.air build/mlx_attn.air build/attn_paged.air build/mlx_gemm.air build/mlx_gemm_splitk.air build/fnp_gather.air build/pf8.air build/qa8.air -o build/core.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/decode_test.mm -o build/decode_test
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/server.mm -o build/inferd
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/spec_test.mm -o build/spec_test
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/kv_check.mm -o build/kv_check
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/qsa_bench.mm -o build/qsa_bench
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/fn_gemm.mm -o build/decode_test_fngemm
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/qsa_stats.mm -o build/decode_test_qsastats
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/qa8_dbg.mm -o build/qa8_dbg  # QA8 kernel vs CPU
clang++ -std=c++20 -O2 tools/tree_sim.cc -o build/tree_sim
clang++ -std=c++20 -O2 tools/grammar_test.cc -o build/grammar_test  # host-only: src/grammar.h (structured output)
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/spec_sample_test.mm -o build/spec_sample_test  # GPU, no model: speculative sampler vs src/sampling.h
# Host-only tests (Codex review 2026-11-02, S07/S08): strict JSON reader, safetensors header validation, SSD tier.
# weights_test also runs every loader family on malformed-but-self-consistent checkpoints (2026-11-03, R01; it needs
# build/core.metallib, built above) and spill_test failed writes (R05). Under ASan/UBSan: see each test's header.
clang++ -std=c++20 -O2 -Isrc tools/json_test.cc -o build/json_test
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/weights_test.mm -o build/weights_test
clang++ -std=c++20 -O2 -Isrc tools/spill_test.cc -o build/spill_test
