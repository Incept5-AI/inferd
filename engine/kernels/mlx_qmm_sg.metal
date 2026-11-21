// MLX's simdgroup-matrix quantized matmul (affine_qmm_t, vendored from third_party/mlx quantized.h, flip-aware via
// function constant 40), for GPUs without neural accelerators (M3/M4): the NAX qmm is wrong there and W4A8's matmul2d
// int8 path is emulated. affine, group 64, bfloat, transposed weights, unbatched, BM = BN = BK = 32, 4 simdgroups.
// Name: mlx_qmm_sg_b<bits>_al<aligned N>.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define QMMSG(bits, al) \
  instantiate_kernel("mlx_qmm_sg_b" #bits "_al" #al, affine_qmm_t, bfloat16_t, 64, bits, al, false)
QMMSG(4, true) QMMSG(4, false) QMMSG(8, true) QMMSG(8, false)

// Tile-shape variants (same kernel, BM x BK x BN template args; WM = WN = 2 simdgroups, so BM, BN are multiples of
// 16; BK <= the group size, 64).
// Name: mlx_qmm_sg_b<bits>_al<aligned N to BN>_<BM>x<BK>x<BN>.
#define QMMSGT(bits, al, bm, bk, bn) \
  instantiate_kernel("mlx_qmm_sg_b" #bits "_al" #al "_" #bm "x" #bk "x" #bn, affine_qmm_t, bfloat16_t, 64, bits, al, \
                     false, false, bm, bk, bn)
#define QMMSGT_AL(bm, bk, bn) QMMSGT(4, true, bm, bk, bn) QMMSGT(4, false, bm, bk, bn) \
  QMMSGT(8, true, bm, bk, bn) QMMSGT(8, false, bm, bk, bn)
// Kept: the shapes Qwen35::sg_tile picks (M3 Ultra sweep, 2026-11-01; BK = 32 and BM = 64 lost everywhere). With
// WM = WN = 2, BM * BK / 128 must divide BK into rows that divide BM, so BM is 16 or 32 at BK 64.
QMMSGT_AL(16, 64, 16) QMMSGT_AL(16, 64, 32) QMMSGT_AL(32, 64, 16) QMMSGT_AL(32, 64, 32)  // BM = 48 here drops x columns (MLX loader)

// MLX's affine_qmv_wide (what MLX itself runs for 2 <= M < ~13-32 rows on gen-15+ GPUs): each threadgroup streams
// NV input vectors and decodes each weight group once for all of them; k_lanes lanes reduce K per output row.
// grid (ceil(M / NV), ceil(N / (2 * 32 / KL))), threads (32, 2). Name: mlx_qmv_wide_b<bits>_nv<NV>_kl<KL>.
#define QMVW(bits, nv, kl) \
  instantiate_kernel("mlx_qmv_wide_b" #bits "_nv" #nv "_kl" #kl, affine_qmv_wide, bfloat16_t, 64, bits, nv, kl, false)
#define QMVW_ALL(bits, kl) QMVW(bits, 1, kl) QMVW(bits, 2, kl) QMVW(bits, 3, kl) QMVW(bits, 4, kl) \
  QMVW(bits, 5, kl) QMVW(bits, 6, kl) QMVW(bits, 7, kl) QMVW(bits, 8, kl)
QMVW_ALL(4, 8) QMVW_ALL(4, 16) QMVW_ALL(8, 8) QMVW_ALL(8, 16)

// qmm_t_impl with the simdgroup layout as template args (MLX fixes WM = WN = 2, so BM is a multiple of 16). WM = 1,
// WN = 4 allows BM in steps of 8: the GEMM is MMA-bound here (cost ~ 20 ms + 2.2 ms per tile row per 27B step on M3
// Ultra), so rounding M up to 8 instead of 16 or 32 is the gain. Same loaders and BlockMMA as MLX; flip-aware via the
// patched dequantize (function constant 40); buffer 8 = the weight row length, so it also serves K-slice views (the ANE
// split's down projection). Name: mlx_qmm_sgw_b<bits>_al<aligned N to BN>_<BM>x<BN>, BK = 64.
// The ordinary path rounds the projection to bf16 before adding it to the residual.
struct SgRoundedResidual {
  static float apply(float product, bfloat16_t residual) {
    return float(bfloat16_t(product)) + float(residual);
  }
};

template <typename T, const int group_size, const int bits, const bool aligned_N, const int BM, const int BK,
          const int BN, const int WM, const int WN, int residual_k = 0, int residual_n = 0>
[[kernel]] void qmm_t_wmwn(
    const device uint32_t* w [[buffer(0)]], const device T* scales [[buffer(1)]], const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]], device T* y [[buffer(4)]], const constant int& input_k [[buffer(5)]],
    const constant int& output_n [[buffer(6)]], const constant int& rows [[buffer(7)]],
    const constant int& weight_stride [[buffer(8)]],  // weight row length in elements: K, or the full K of a K-slice view
    uint3 tid [[threadgroup_position_in_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int K = residual_k ? residual_k : input_k, N = residual_n ? residual_n : output_n;
  const int M = residual_k ? BM : rows, ldw = residual_k ? residual_k : weight_stride;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  // BlockLoader's default reads BM * BK / 128 values per thread, which leaves columns unloaded unless that divides BK
  // into a row count dividing BM (BM = 24/40/48 at BK 64 silently dropped 4-16 columns of x). 4 per thread tiles
  // any BM that is a multiple of 8.
  constexpr short nrd = BM * BK / (WM * WN * SIMD_SIZE);
  constexpr short nr = (BK % nrd == 0 && BM % ((WM * WN * SIMD_SIZE) / (BK / nrd)) == 0) ? nrd : 4;
  static_assert(BM % 8 == 0 && BK % 64 == 0, "qmm_t_wmwn: BM multiple of 8, BK 64");
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, nr>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = ldw * bytes_per_pack / pack_factor;
  const int K_g = ldw / group_size;
  // The fixed-shape residual dispatch covers all 48 rows with one tile vertically.
  // Making that explicit also lets the compiler omit the partial-row loader.
  const int y_row = residual_k ? 0 : int(tid.y) * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, ldw, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  const bool full_m = num_els == BM, full_n = aligned_N || num_outs == BN;
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full_m) loader_x.load_unsafe(); else loader_x.load_safe(short2(BK, num_els));
    if (full_n) loader_w.load_unsafe(); else loader_w.load_safe(short2(BK, num_outs));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  if constexpr (!residual_k) threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (residual_k != 0) mma_op.apply_epilogue(y, N, 1, SgRoundedResidual{});
  if (!full_m || !full_n) mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  else mma_op.store_result(y, N);
}

#define QMMSGW(bits, al, bm, bn) \
  template [[host_name("mlx_qmm_sgw_b" #bits "_al" #al "_" #bm "x" #bn)]] [[kernel]] void \
  qmm_t_wmwn<bfloat16_t, 64, bits, al, bm, 64, bn, 1, 4>( \
      const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, const device bfloat16_t*, \
      device bfloat16_t*, const constant int&, const constant int&, const constant int&, const constant int&, uint3, \
      uint, uint);
#define QMMSGW_AL(bm, bn) QMMSGW(4, true, bm, bn) QMMSGW(4, false, bm, bn) QMMSGW(8, true, bm, bn) QMMSGW(8, false, bm, bn)
QMMSGW_AL(8, 32) QMMSGW_AL(16, 32) QMMSGW_AL(24, 32) QMMSGW_AL(32, 32) QMMSGW_AL(40, 32) QMMSGW_AL(48, 32)
QMMSGW_AL(56, 32)  // 64x32: 442 ms per 27B step at 64 rows against 406 for the 2x2 32x64x32 above (M3 Max)

template [[host_name("mlx_qmm_sgw_down48_residual")]] [[kernel]] void
qmm_t_wmwn<bfloat16_t, 64, 4, true, 48, 64, 32, 1, 4, 17408, 5120>(
    const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, const device bfloat16_t*,
    device bfloat16_t*, const constant int&, const constant int&, const constant int&, const constant int&, uint3,
    uint, uint);

template [[host_name("mlx_qmm_sgw_out48_residual")]] [[kernel]] void
qmm_t_wmwn<bfloat16_t, 64, 4, true, 48, 64, 32, 1, 4, 6144, 5120>(
    const device uint32_t*, const device bfloat16_t*, const device bfloat16_t*, const device bfloat16_t*,
    device bfloat16_t*, const constant int&, const constant int&, const constant int&, const constant int&, uint3,
    uint, uint);

// A separate unpack lets all row tiles share the same BF16 weights in a large prefill.
// dequantize uses the same affine BF16 arithmetic and flip flag as the packed simdgroup path.
kernel void prefill_unpack4(device const uchar* packed [[buffer(0)]], device const bfloat* scales [[buffer(1)]],
                            device const bfloat* biases [[buffer(2)]], device bfloat* weights [[buffer(3)]],
                            uint pack [[thread_position_in_grid]]) {
  dequantize<bfloat, 8, 4>(packed + 4 * pack, scales[pack / 8], biases[pack / 8], weights + 8 * pack);
}

kernel void prefill_dense_gdn(device const bfloat* input [[buffer(0)]], device const bfloat* weights [[buffer(1)]],
                              device bfloat* output [[buffer(2)]], constant int& M [[buffer(3)]],
                              constant int& N [[buffer(4)]], constant int& K [[buffer(5)]],
                              uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64, BN = 64, BK = 32, stride = BK + 8;
  threadgroup bfloat xs[BM * stride], ws[BN * stride];
  using XLoader = mlx::steel::BlockLoader<bfloat, BM, BK, stride, 1, 128, 1, 4>;
  using WLoader = mlx::steel::BlockLoader<bfloat, BN, BK, stride, 1, 128, 1, 4>;
  using MMA = mlx::steel::BlockMMA<bfloat, bfloat, BM, BN, BK, 2, 2, false, true, stride, stride>;
  const int row = group.y * BM, col = group.x * BN;
  const short rows = min(BM, M - row), cols = min(BN, N - col);
  XLoader x(input + size_t(row) * K, K, xs, simd, lane);
  WLoader w(weights + size_t(col) * K, K, ws, simd, lane);
  MMA mma(simd, lane);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rows == BM) x.load_unsafe(); else x.load_safe(short2(BK, rows));
    if (cols == BN) w.load_unsafe(); else w.load_safe(short2(BK, cols));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(xs, ws);
    x.next();
    w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  output += size_t(row) * N + col;
  if (rows == BM && cols == BN) mma.store_result(output, N);
  else mma.store_result_safe(output, N, short2(cols, rows));
}

// Two workgroups share one 48-row down-projection output tile. Each computes half of K
// into the existing FP32 split-K workspace. The change in summation order can affect greedy output.
kernel void sg_down48_split2(device const uint8_t* packed [[buffer(0)]], device const bfloat* scales [[buffer(1)]],
                             device const bfloat* biases [[buffer(2)]], device const bfloat* input [[buffer(3)]],
                             device float* partial [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int M = 48, N = 5120, K = 17408, BN = 32, BK = 64, stride = BK + 8, slice = K / 2;
  threadgroup bfloat xs[M * stride], ws[BN * stride];
  using XLoader = mlx::steel::BlockLoader<bfloat, M, BK, stride, 1, 128, 1, 4>;
  using WLoader = QuantizedBlockLoader<bfloat, BN, BK, stride, 1, 128, 64, 4>;
  using MMA = mlx::steel::BlockMMA<bfloat, float, M, BN, BK, 2, 2, false, true, stride, stride>;
  const int col = group.x * BN, start = group.z * slice;
  const size_t offset = size_t(col) * K + start;
  XLoader x(input + start, K, xs, sg, lane);
  WLoader w(packed + offset / 2, scales + offset / 64, biases + offset / 64, K, ws, sg, lane);
  MMA mma(sg, lane);
  for (int k = 0; k < slice; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x.load_unsafe();
    w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(xs, ws);
    x.next();
    w.next();
  }
  mma.store_result(partial + group.z * M * N + col, N);
}

kernel void sg_down48_split2_residual(device const float* partial [[buffer(0)]], device bfloat* residual [[buffer(1)]],
                                      uint i [[thread_position_in_grid]]) {
  constexpr int elements = 48 * 5120;
  const float sum = partial[i] + partial[elements + i];
  residual[i] = bfloat(SgRoundedResidual::apply(sum, residual[i]));
}

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
using namespace metal;
using namespace mlx::steel;
using EvFrag=BaseMMAFrag<float,8,8>;
struct EvPackParams {int k,n,flip,slab,dense;};
kernel void ev_pack_weights(device const uchar* w [[buffer(0)]],device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],device ulong* codes [[buffer(3)]],device bfloat2* meta [[buffer(4)]],
    constant EvPackParams& p [[buffer(5)]], uint i [[thread_position_in_grid]]){
    const int lane=i%32,group=(i/32/p.slab)%(p.k/64);
    const int frag=(i/(32*p.slab*(p.k/64)))*p.slab+(i/32)%p.slab;
    const short2 c=EvFrag::get_coord(lane);
    const int col=min(frag*8+c.x,p.n-2),a=col*(p.k/64)+group,b=a+p.k/64;
    const float2 sc=float2(scales[a],scales[b]),bi=float2(biases[a],biases[b]);
    ulong packet=0;
    for(int s=0;s<8;++s){
      int k=group*64+8*s+c.y;const uint shift=(k%2)*4;
      uint2 v=(uint2(w[(col*p.k+k)/2],w[((col+1)*p.k+k)/2])>>shift)&15;
      v^=p.flip?8u:0u;
      if(p.dense)reinterpret_cast<device bfloat2*>(codes)[(i/32*8+s)*32+lane]=bfloat2(float2(v)*sc+bi);
      else packet|=ulong(v.x|(v.y<<4))<<(8*s);
    }
    if(!p.dense)codes[i]=packet;
    if(lane<8 && !p.dense){
      const int column=min(frag*8+lane,p.n-1),j=column*(p.k/64)+group;
      meta[i/32*8+lane]=bfloat2(scales[j],biases[j]);
    }
}
template<int K,int N,int BN,int WM,int WN,int Slab,bool Dense,bool Native,bool Mixed>
kernel void ev_fragment_matmul(device const uint* weights [[buffer(0)]],device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],device const bfloat* input [[buffer(3)]],device bfloat* output [[buffer(4)]],
    constant int2& flags [[buffer(5)]],uint block [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]){
    constexpr int TM=48/(8*WM),TN=BN/(8*WN);
    const short2 xy=EvFrag::get_coord(lane);const int row=(sg/WN)*8+xy.y,c0=block*BN+(sg%WN)*8;
    MMATile<float,TM,TN> total;total.clear();MMATile<float,TM,1> a;MMATile<float,1,TN> b;
    for(int g=0;g<K/64;g++){
      float2 sc[TN],bi[TN];ulong packet[TN];int addr[TN];
      for(int n=0;n<TN;n++){
        int col=min(c0+n*WN*8+xy.x,N-2),frag=(c0+n*WN*8)/8;
        int index=(frag/Slab*(K/64)+g)*Slab+frag%Slab;addr[n]=index;
        if constexpr(Native){
          int j=col*(K/64)+g;sc[n]=float2(scales[j],scales[j+K/64]);bi[n]=float2(biases[j],biases[j+K/64]);
        }else if constexpr(!Dense){
          packet[n]=reinterpret_cast<device const ulong*>(weights)[index*32+lane];
          auto m=reinterpret_cast<device const bfloat2*>(scales)+index*8+xy.x;
          sc[n]=float2(m[0].x,m[1].x);bi[n]=float2(m[0].y,m[1].y);
        }
      }
      for(int sub=0;sub<8;sub++){
        for(int m=0;m<TM;m++)a.frag_at(m,0)=float2(*reinterpret_cast<device const bfloat2*>(input+(row+m*WM*8)*K+g*64+8*sub+xy.x));
        for(int n=0;n<TN;n++){
          if constexpr(Dense)b.frag_at(0,n)=float2(reinterpret_cast<device const bfloat2*>(weights)[(addr[n]*8+sub)*32+lane]);
          else{
            uint2 v;
            if constexpr(Native){
              int col=min(c0+n*WN*8+xy.x,N-2),j=col*(K/8)+8*g+sub;
              v=((uint2(weights[j],weights[j+K/8])>>(4*xy.y))&15u)^uint(flags.x?8:0);
            }else{uint bytes=uint(packet[n]>>(8*sub));v=uint2(bytes,bytes>>4)&15u;}
            b.frag_at(0,n)=float2(bfloat2(float2(v)*sc[n]+bi[n]));
          }
        }
        if constexpr(Mixed){
          for(int m=0;m<TM;++m)for(int n=0;n<TN;++n){
            simdgroup_half8x8 am,bm;simdgroup_float8x8 cm,dm;
            reinterpret_cast<thread half2&>(am.thread_elements())=half2(a.frag_at(m,0));
            reinterpret_cast<thread half2&>(bm.thread_elements())=half2(b.frag_at(0,n));
            reinterpret_cast<thread float2&>(cm.thread_elements())=total.frag_at(m,n);
            simdgroup_multiply_accumulate(dm,am,bm,cm);
            total.frag_at(m,n)=reinterpret_cast<thread float2&>(dm.thread_elements());
          }
        }else tile_matmad(total,a,b,total);
      }
    }
    for(int m=0;m<TM;++m)for(int n=0;n<TN;++n){
      int col=c0+n*WN*8+xy.x;if(N%BN && col>=N)continue;
      auto dst=reinterpret_cast<device bfloat2*>(output+(row+m*WM*8)*N+col);
      bfloat2 v=bfloat2(total.frag_at(m,n));if(flags.y)v=bfloat2(float2(v)+float2(*dst));*dst=v;
    }
}
#define EV_FRAGMENT(K,N,BN,WM,WN,SL,D,A,M,NAME) template [[host_name(NAME)]] kernel void ev_fragment_matmul<K,N,BN,WM,WN,SL,D,A,M>(device const uint*,device const bfloat*,device const bfloat*,device const bfloat*,device bfloat*,constant int2&,uint,uint,uint);
EV_FRAGMENT(5120,34816,64,2,2,1,1,0,0,"evf_5120_34816_2_1_64_2_0")
EV_FRAGMENT(17408,5120,8,3,1,8,1,0,0,"evf_17408_5120_2_8_8_31_0")
EV_FRAGMENT(6144,5120,16,3,2,8,1,0,0,"evf_6144_5120_2_8_16_32_0")
EV_FRAGMENT(5120,14336,64,2,2,1,1,0,0,"evf_5120_14336_2_1_64_2_0")
constant constexpr int CV=248320,CK=5120,CC=512;
struct CertRow{int slot,id;float radius,norm,lower;};
kernel void cert_norm(device const uint* w [[buffer(0)]],device const bfloat* s [[buffer(1)]],device const bfloat* b [[buffer(2)]],
    device float* norms [[buffer(3)]],constant int& flip [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
    threadgroup float p[4];float val=0;
    for(int k=t;k<CK;k+=128){uint q=((w[row*640+k/8]>>(4*(k%8)))&15)^uint(flip?8:0);float v=bfloat(float(q)*float(s[row*80+k/64])+float(b[row*80+k/64]));val+=v*v;}
    val=simd_sum(val);if(!lane)p[sg]=val;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!t)norms[row]=sqrt(p[0]+p[1]+p[2]+p[3])*1.001f+.00001f;
}
kernel void cert_near(device const bfloat* x [[buffer(0)]],device const bfloat* xold [[buffer(1)]],device const int* proposed [[buffer(2)]],
    device const int* ids [[buffer(3)]],device CertRow* picked [[buffer(4)]],constant float& threshold [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint t [[thread_index_in_threadgroup]]){
    threadgroup float ds[8],xs[8],os[8];int slot=proposed[row*8+sg];float err=0,xn=0,on=0;
    if(slot>=0)for(int k=lane;k<CK;k+=32){float v=x[row*CK+k],w=xold[slot*CK+k];err+=(v-w)*(v-w);xn+=v*v;on+=w*w;}
    err=sqrt(simd_sum(err));xn=sqrt(simd_sum(xn));on=sqrt(simd_sum(on));
    if(!lane){ds[sg]=slot<0?INFINITY:err;xs[sg]=xn;os[sg]=on;}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!t){int best=0;for(int j=1;j<8;j++)if(ds[j]<ds[best])best=j;
      int k=ds[best]<threshold?proposed[row*8+best]:-1;
      picked[row]={k,k>=0?ids[k]:0,(ds[best]+.000315f*(xs[best]+os[best]))*1.001f+.0001f,xs[best]*1.001f,0};}
}
kernel void cert_incumbent(device const uint* w [[buffer(0)]],device const bfloat* scales [[buffer(1)]],device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],device CertRow* picked [[buffer(4)]],device const float* wn [[buffer(5)]],
    constant int& flip [[buffer(6)]],device atomic_uint* cnt [[buffer(7)]],uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
    CertRow info=picked[row];if(!t)atomic_store_explicit(cnt+row,info.slot<0?CC+1:0,memory_order_relaxed);if(info.slot<0)return;
    int id=info.id;float val=0;threadgroup float partial[4];
    for(int k=t;k<CK;k+=128){
      uint q=((w[id*640+k/8]>>(4*(k%8)))&15)^uint(flip?8:0);
      val+=float(x[row*CK+k])*float(bfloat(float(q)*float(scales[id*80+k/64])+float(biases[id*80+k/64])));
    }
    val=simd_sum(val);if(!lane)partial[sg]=val;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!t){float score=partial[0]+partial[1]+partial[2]+partial[3];
      picked[row].lower=score-.00064f*wn[id]*info.norm-abs(score)/250.f-.002f;}
}
kernel void cert_filter(device const bfloat* old [[buffer(0)]],device const float* wn [[buffer(1)]],device const CertRow* picked [[buffer(2)]],
    device atomic_uint* cnt [[buffer(3)]],device int* candidates [[buffer(4)]],uint2 t [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    CertRow p=picked[t.y];if(p.slot<0 || atomic_load_explicit(cnt+t.y,memory_order_relaxed)>CC)return;
    float v=old[p.slot*CV+t.x],upper=v+abs(v)/250.f+p.radius*wn[t.x]+.001f;
    bool pass=upper+abs(upper)/250.f+.001f>=p.lower;
    uint total=simd_sum(uint(pass)),offset=simd_prefix_exclusive_sum(uint(pass)),base=0;
    if(!lane && total)base=atomic_fetch_add_explicit(cnt+t.y,total,memory_order_relaxed);
    base=simd_broadcast_first(base);
    if(pass && base+offset<CC)candidates[t.y*CC+base+offset]=t.x;
}
kernel void cert_plan(device const uint* cnt [[buffer(0)]],device int* map [[buffer(1)]],device int* orig [[buffer(2)]],
    device uint* grid [[buffer(3)]],uint t [[thread_position_in_grid]]){
    if(t)return;int full=0,maxcount=0;
    for(int i=0;i<48;++i){if(!cnt[i] || cnt[i]>CC){map[i]=full;orig[full++]=i;}else{map[i]=-1;maxcount=max(maxcount,int(cnt[i]));}}
    for(int i=full;i<48;++i)orig[i]=0;
    for(int k=1;k<=6;k++){grid[3*(k-1)]=k==(full+7)/8?CV/32:0;grid[3*(k-1)+1]=grid[3*(k-1)+2]=1;}
    grid[18]=maxcount>1?(maxcount+7)/8:0;grid[19]=48;grid[20]=1;
}
kernel void cert_gather(device const bfloat2* x [[buffer(0)]],device bfloat2* dst [[buffer(1)]],device const int* orig [[buffer(2)]],uint2 t [[thread_position_in_grid]]){
    dst[t.y*(CK/2)+t.x]=x[orig[t.y]*(CK/2)+t.x];
}
kernel void cert_short(device const uint* w [[buffer(0)]],device const bfloat* s [[buffer(1)]],device const bfloat* b [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],device const uint* count [[buffer(4)]],device const int* cand [[buffer(5)]],
    device bfloat* val [[buffer(6)]],constant int& flip [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    uint row=group.y,off=group.x*8,ct=count[row];if(ct<=1 || ct>CC || off>=ct)return;
    short2 xy=EvFrag::get_coord(lane);int c=off+xy.x,id=cand[row*CC+min(uint(c),ct-1)],id2=cand[row*CC+min(uint(c+1),ct-1)];
    MMATile<float,1,1> total,a,bt;total.clear();
    for(int g=0;g<CK/64;g++){
      float2 scale={float(s[id*80+g]),float(s[id2*80+g])},bias={float(b[id*80+g]),float(b[id2*80+g])};
      for(int h=0;h<8;h++){
        a.frag_at(0,0)=float2(*reinterpret_cast<device const bfloat2*>(x+row*CK+64*g+8*h+xy.x));
        uint2 code=((uint2(w[id*640+8*g+h],w[id2*640+8*g+h])>>(4*xy.y))&15u)^uint(flip?8:0);
        bt.frag_at(0,0)=float2(bfloat2(float2(code)*scale+bias));tile_matmad(total,a,bt,total);
      }
    }
    if(!xy.y){auto v=bfloat2(total.frag_at(0,0));if(uint(c)<ct)val[row*CC+c]=v.x;if(uint(c+1)<ct)val[row*CC+c+1]=v.y;}
}
kernel void cert_finish(device const bfloat* whole [[buffer(0)]],device const uint* cnt [[buffer(1)]],device const int* fmap [[buffer(2)]],
    device const int* cand [[buffer(3)]],device const bfloat* val [[buffer(4)]],device int* pred [[buffer(5)]],
    device int* remembered [[buffer(6)]],constant int& write [[buffer(7)]],uint row [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
    int map=fmap[row],n=map>=0?CV:int(cnt[row]);if(n==1){if(!t)pred[row]=cand[row*CC];return;}
    threadgroup float vals[32];threadgroup int ids[32];float best=-INFINITY;int id=INT_MAX;
    for(int j=t;j<n;j+=1024){float v=map>=0?whole[map*CV+j]:val[row*CC+j];int k=map>=0?j:cand[row*CC+j];if(v>best || (v==best && k<id)){best=v;id=k;}}
    float top=simd_max(best);id=simd_min(best==top?id:INT_MAX);
    if(!lane){vals[sg]=top;ids[sg]=id;}threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){float v=vals[lane],mx=simd_max(v);int key=simd_min(v==mx?ids[lane]:INT_MAX);if(!lane){pred[row]=key;if(map>=0)remembered[write+row]=key;}}
}
kernel void cert_store(device const bfloat2* whole [[buffer(0)]],device const bfloat2* x [[buffer(1)]],device const int* map [[buffer(2)]],
    device bfloat2* old [[buffer(3)]],device bfloat2* ox [[buffer(4)]],constant int& slot [[buffer(5)]],uint2 t [[thread_position_in_grid]]){
    int j=map[t.y];if(j<0)return;
    old[(slot+t.y)*(CV/2)+t.x]=whole[j*(CV/2)+t.x];
    if(t.x<CK/2)ox[(slot+t.y)*(CK/2)+t.x]=x[t.y*(CK/2)+t.x];
}
kernel void cert_block_norm(device const float* wn [[buffer(0)]],device float* maxnorm [[buffer(1)]],
    uint block [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup float s[8];float v=simd_max(wn[block*256+t]);if(!lane)s[sg]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){float top=simd_max(lane<8?s[lane]:0.f);if(!lane)maxnorm[block]=top;}
}
kernel void cert_reset(device uint* grid [[buffer(0)]],uint t [[thread_position_in_grid]]){grid[t]=t?1:0;}
kernel void cert_blocks(device const bfloat* blockmax [[buffer(0)]],device const float* norms [[buffer(1)]],device const CertRow* picked [[buffer(2)]],
    device atomic_uint* ng [[buffer(3)]],device uint* jobs [[buffer(4)]],uint2 t [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    constexpr uint nb=CV/256;CertRow p=picked[t.y];bool ok=false;
    if(t.x<nb && p.slot>=0){float v=blockmax[p.slot*nb+t.x],u=v+abs(v)/250.f+p.radius*norms[t.x]+.001f;ok=u+abs(u)/250.f+.001f>=p.lower;}
    uint n=simd_sum(uint(ok)),idx=simd_prefix_exclusive_sum(uint(ok)),offset=0;
    if(!lane && n)offset=atomic_fetch_add_explicit(ng,n,memory_order_relaxed);offset=simd_broadcast_first(offset);
    if(ok)jobs[offset+idx]=t.y*nb+t.x;
}
kernel void cert_filter_blocks(device const bfloat* old [[buffer(0)]],device const float* wn [[buffer(1)]],device const CertRow* picked [[buffer(2)]],
    device atomic_uint* cnt [[buffer(3)]],device int* candidates [[buffer(4)]],device const uint* jobs [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    constexpr int nb=CV/256;uint job=jobs[group],row=job/nb,col=job%nb*256+t;
    if(simd_broadcast_first(atomic_load_explicit(cnt+row,memory_order_relaxed))>CC)return;
    CertRow p=picked[row];float v=old[p.slot*CV+col],u=v+abs(v)/250.f+p.radius*wn[col]+.001f;
    bool ok=u+abs(u)/250.f+.001f>=p.lower;uint n=simd_sum(uint(ok)),idx=simd_prefix_exclusive_sum(uint(ok)),offset=0;
    if(!lane && n)offset=atomic_fetch_add_explicit(cnt+row,n,memory_order_relaxed);offset=simd_broadcast_first(offset);
    if(ok && offset+idx<CC)candidates[row*CC+offset+idx]=col;
}
kernel void cert_store_blocks(device const bfloat2* whole [[buffer(0)]],device const bfloat2* x [[buffer(1)]],device const int* map [[buffer(2)]],
    device bfloat2* old [[buffer(3)]],device bfloat2* ox [[buffer(4)]],constant int& slot [[buffer(5)]],device bfloat* tops [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    int row=group.y,j=map[row];if(j<0)return;
    uint col=group.x*128+t;auto pair=whole[j*(CV/2)+col];old[(slot+row)*(CV/2)+col]=pair;
    if(col<CK/2)ox[(slot+row)*(CK/2)+col]=x[row*(CK/2)+col];
    threadgroup float s[4];float v=simd_max(max(float(pair.x),float(pair.y)));
    if(!lane)s[sg]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){float top=simd_max(lane<4?s[lane]:-INFINITY);if(!lane)tops[(slot+row)*(CV/256)+group.x]=bfloat(top);}
}
