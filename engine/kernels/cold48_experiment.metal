#include <metal_stdlib>
using namespace metal;
// One simdgroup per value head. Save a compact key and decay, then project old
// updates onto the current key and query just once for all 128 value rows.
struct LogArgs { int phase, interval; };
kernel void log_coeff(device const bfloat* co [[buffer(0)]], device const float* decay [[buffer(1)]],
                      device const int* slots [[buffer(2)]], device bfloat4* keys [[buffer(3)]],
                      device float* gates [[buffer(4)]], device float4* coeff [[buffer(5)]],
                      constant LogArgs& p [[buffer(6)]], uint h [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    int row=h/48, vh=h%48, kh=vh/3;
    int base=(slots[row]*48+vh)*p.interval;
    device const bfloat4* qkv=(device const bfloat4*)(co+row*10240);
    float4 q=float4(qkv[kh*32+lane]), k=float4(qkv[(kh+16)*32+lane]);
    float g=decay[h], product=g;
    for(int j=p.phase-1;j>=0;--j){
        float4 old=float4(keys[(base+j)*32+lane]);
        float ko=simd_sum(dot(old,k)), qo=simd_sum(dot(old,q));
        if(!lane) coeff[h*17+j]=float4(product*ko,product*qo,product,0);
        product*=gates[base+j];
    }
    float qk=simd_sum(dot(q,k));
    if(!lane){coeff[h*17+16]=float4(product,qk,0,0);gates[base+p.phase]=g;}
    keys[(base+p.phase)*32+lane]=bfloat4(k);
}
template<bool Merge>
kernel void log_update(device const bfloat* co [[buffer(0)]],device const float* beta [[buffer(1)]],
                       device float4* state [[buffer(2)]],device bfloat* y [[buffer(3)]],
                       device const int* slots [[buffer(4)]],device const bfloat4* keys [[buffer(5)]],
                       device float* deltas [[buffer(6)]],device const float4* coeff [[buffer(7)]],
                       constant LogArgs& p [[buffer(8)]],uint3 gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    int h=gid.z,row=h/48,vh=h%48,kh=vh/3,sh=slots[row]*48+vh;
    int base=sh*p.interval, dv=gid.y*4;
    device const bfloat4* qkv=(device const bfloat4*)(co+row*10240);
    float4 k=float4(qkv[(16+kh)*32+lane]),q=float4(qkv[kh*32+lane]);
    float4 cq=coeff[h*17+16], out, delta;
    float4 s[4], qs,ks;
    for(int r=0;r<4;++r){
        s[r]=state[(sh*128+dv+r)*32+lane];
        ks[r]=simd_sum(dot(s[r],k))*cq.x; qs[r]=simd_sum(dot(s[r],q))*cq.x;
        if(Merge)s[r]*=cq.x;
    }
    for(int j=0;j<p.phase;++j){
        float4 d=*((device const float4*)(deltas+(base+j)*128+dv));
        float4 c=coeff[h*17+j];
        ks+=d*c.x; qs+=d*c.y;
        if(Merge){
            float4 old=float4(keys[(base+j)*32+lane])*c.z;
            for(int r=0;r<4;++r)s[r]+=old*d[r];
        }
    }
    float4 v=float4(qkv[(4096+vh*128+dv)/4]);
    delta=(v-ks)*beta[h]; out=qs+delta*cq.y;
    if(!lane){
        *((device bfloat4*)(y+h*128+dv))=bfloat4(out);
        if(!Merge)*((device float4*)(deltas+(base+p.phase)*128+dv))=delta;
    }
    if(Merge) for(int r=0;r<4;++r)state[(sh*128+dv+r)*32+lane]=s[r]+k*delta[r];
}
#define INST(M,N) template [[host_name(N)]] kernel void log_update<M>(device const bfloat*,device const float*,device float4*,device bfloat*,device const int*,device const bfloat4*,device float*,device const float4*,constant LogArgs&,uint3,uint);
INST(false,"log_append")
INST(true,"log_merge")
// Replay the original FP32 operations in registers. Store the full state only on merge.
template<bool Merge>
kernel void exact_deferred(device const bfloat* co [[buffer(0)]],device const float* gate [[buffer(1)]],device const float* beta [[buffer(2)]],
                           device float* state [[buffer(3)]],device bfloat* out [[buffer(4)]],device const int* slots [[buffer(5)]],
                           device bfloat4* keys [[buffer(6)]],device float* decays [[buffer(7)]],device float* deltas [[buffer(8)]],
                           constant LogArgs& p [[buffer(9)]],uint3 gid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    int h=gid.z,b=h/48,vh=h%48,kh=vh/3,dv=4*gid.y,sh=slots[b]*48+vh,base=sh*p.interval;
    device float* st=state+(sh*128+dv)*128;
    device const bfloat* row=co+b*10240;
    float s[4][4],k[4],q[4],kv[4],o[4],delta[4];
    for(int i=0;i<4;++i){k[i]=row[(16+kh)*128+4*lane+i];q[i]=row[kh*128+4*lane+i];}
    for(int r=0;r<4;++r)for(int i=0;i<4;++i)s[r][i]=st[r*128+4*lane+i];
    for(int j=0;j<p.phase;++j){
        float4 old=float4(keys[(base+j)*32+lane]);
        float gg=decays[base+j];
        float4 d=*((device const float4*)(deltas+(base+j)*128+dv));
        for(int r=0;r<4;++r)for(int i=0;i<4;++i){s[r][i]*=gg;s[r][i]+=old[i]*d[r];}
    }
    for(int r=0;r<4;++r){
        kv[r]=0;for(int i=0;i<4;++i){s[r][i]*=gate[h];kv[r]+=s[r][i]*k[i];}
    }
    for(int r=0;r<4;++r)kv[r]=simd_sum(kv[r]);
    for(int r=0;r<4;++r){
        delta[r]=(float(row[4096+vh*128+dv+r])-kv[r])*beta[h];
        o[r]=0;for(int i=0;i<4;++i){s[r][i]+=k[i]*delta[r];o[r]+=s[r][i]*q[i];}
    }
    for(int r=0;r<4;++r)o[r]=simd_sum(o[r]);
    if(lane<4){out[h*128+dv+lane]=bfloat(o[lane]);if(!Merge)deltas[(base+p.phase)*128+dv+lane]=delta[lane];}
    if(Merge){for(int r=0;r<4;++r)for(int i=0;i<4;++i)st[r*128+4*lane+i]=s[r][i];}
    else if(dv==0){keys[(base+p.phase)*32+lane]=*((device const bfloat4*)(row+(16+kh)*128+4*lane));if(!lane)decays[base+p.phase]=gate[h];}
}
#define EXACT(M,N) template [[host_name(N)]] kernel void exact_deferred<M>(device const bfloat*,device const float*,device const float*,device float*,device bfloat*,device const int*,device bfloat4*,device float*,device float*,constant LogArgs&,uint3,uint);
EXACT(false,"exact_append")
EXACT(true,"exact_merge")

kernel void flush_logs(device float4* state [[buffer(0)]], device const bfloat4* keys [[buffer(1)]],
                       device const float* gates [[buffer(2)]],device const float* deltas [[buffer(3)]],
                       device const int* slots [[buffer(4)]],constant LogArgs& p [[buffer(5)]],
                       uint3 t [[thread_position_in_grid]]){
    int sh=slots[t.z/48]*48+t.z%48,base=sh*p.interval,idx=(sh*128+t.y)*32+t.x;
    float4 s=state[idx];
    for(int j=0;j<p.phase;++j){
        s*=gates[base+j];s+=float4(keys[(base+j)*32+t.x])*deltas[(base+j)*128+t.y];
    }
    state[idx]=s;
}

#include <metal_stdlib>
using namespace metal;
struct PrefixArgs { int length,ns,max_blocks; };
kernel void prefix_kv(device const char4* k [[buffer(0)]],device const char4* v [[buffer(1)]],
                      device const float* ks [[buffer(2)]],device const float* vs [[buffer(3)]],
                      device const int* table [[buffer(4)]],device half4* pk [[buffer(5)]],device half4* pv [[buffer(6)]],
                      constant PrefixArgs& p [[buffer(7)]],uint3 t [[thread_position_in_grid]]){
    int s=t.z*p.ns+table[t.y/32]*32+t.y%32, out=(t.z*p.length+t.y)*64+t.x;
    pk[out]=half4(float4(k[s*64+t.x])*ks[s]);pv[out]=half4(float4(v[s*64+t.x])*vs[s]);
}
kernel void prefix_q(device const bfloat4* q [[buffer(0)]],device half4* out [[buffer(1)]],
                     device const int* pos [[buffer(2)]],device int* tail [[buffer(3)]],constant int& n [[buffer(4)]],
                     uint3 t [[thread_position_in_grid]]){
    int kh=t.z/6, qh=t.z%6;
    out[((kh*48+t.y)*6+qh)*64+t.x]=half4(q[(t.y*24+t.z)*64+t.x]);
    if(t.x==0 && t.z==0) tail[t.y]=pos[t.y]-n;
}
kernel void prefix_softmax(device const half* scores [[buffer(0)]], device half* probs [[buffer(1)]],
                           device float* stats [[buffer(2)]],constant int& n [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    device const half* s=scores+row*n;device half* p=probs+row*n;
    threadgroup float tmp[8]; float m=-INFINITY;
    for(int j=tid;j<n;j+=256)m=max(m,float(s[j]));
    m=simd_max(m);if(lane==0)tmp[sg]=m;threadgroup_barrier(mem_flags::mem_threadgroup);
    m=simd_max(lane<8?tmp[lane]:-INFINITY);threadgroup_barrier(mem_flags::mem_threadgroup);
    float z=0;for(int j=tid;j<n;j+=256)z+=exp(float(s[j])-m);
    z=simd_sum(z);if(lane==0)tmp[sg]=z;threadgroup_barrier(mem_flags::mem_threadgroup);
    z=simd_sum(lane<8?tmp[lane]:0.f);
    if(tid==0)stats[row]=m+log(z);
    for(int j=tid;j<n;j+=256)p[j]=half(exp(float(s[j])-m)/z);
}
kernel void prefix_partial(device const half* out [[buffer(0)]],device const float* stats [[buffer(1)]],
                           device float* part [[buffer(2)]],uint3 t [[thread_position_in_grid]]){
    int src=(t.z/6*48+t.y)*6+t.z%6, dst=((t.y*24+t.z)*2+1)*258;
    part[dst+t.x]=out[src*256+t.x];
    if(t.x==0){part[dst+256]=stats[src];part[dst+257]=1;}
}
