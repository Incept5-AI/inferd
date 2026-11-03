# Baseline: MLX quantized_matmul (affine 4-bit, group 64) at the shapes qtest uses.
import time, mlx.core as mx
N, K = 17408, 5120
w = mx.random.normal((N, K)).astype(mx.float16)
for bits in (4, 8):
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    for M in (1, 16, 32, 64, 128, 256, 1024):
        x = mx.random.normal((M, K)).astype(mx.float16)
        f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
        for _ in range(3): mx.eval(f())
        ts = []
        for _ in range(5):
            t = time.perf_counter(); mx.eval(*[f() for _ in range(20)]); ts.append((time.perf_counter() - t) / 20)
        t = sorted(ts)[2]
        print(f"mlx q{bits} M={M:5d} {t*1e3:8.4f} ms {2*M*N*K/t/1e12:6.1f} TFLOPS")
