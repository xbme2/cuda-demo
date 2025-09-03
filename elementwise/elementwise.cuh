#include "cuda_header.h"

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
void add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_v0(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_vf(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec4_idx = idx * 4;
    if (vec4_idx + 3 < N) {
        float4 va = reinterpret_cast<float4 *>(a)[idx];
        float4 vb = reinterpret_cast<float4 *>(b)[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        reinterpret_cast<float4 *>(c)[idx] = vc;
    }
}

void relu_cpu(float *i_data, float *o_data, int N) {
    for (int i = 0; i < N; i++) {
        o_data[i] = fmaxf(i_data[0], 0.f);
    }
}

void relu_v0(float *i_data, float *o_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        o_data[idx] = fmaxf(i_data[idx], 0.f);
    }
}

void relu_vf(float *i_data, float *o_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec4_idx = idx * 4;
    if (vec4_idx + 3 < N) {
        float4 i4_data = reinterpret_cast<float4 *>(i_data)[idx];
        float4 o4_data;
        o4_data.x = fmax(i4_data.x, 0.f);
        o4_data.y = fmax(i4_data.y, 0.f);
        o4_data.z = fmax(i4_data.z, 0.f);
        o4_data.w = fmax(i4_data.w, 0.f);

        reinterpret_cast<float4 *>(o_data)[idx] = o4_data;
    }
}

void sigmoid_cpu(float *i_data, float *o_data, int N) {
    for (int i = 0; i < N; i++) {
        o_data[i] = 1.f / (1.f + expf(-i_data[i]));
    }
}

void sigmoid_v0(float *i_data, float *o_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        o_data[idx] = 1.f / (1.f + expf(-i_data[idx]));
    }
}

void sigmoid_vf(float *i_data, float *o_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec4_idx = idx * 4;
    if (vec4_idx + 3 < N) {
        float4 i4_data = reinterpret_cast<float4 *>(i_data)[idx];
        float4 o4_data;
        o4_data.x = 1.f / (1.f + expf(-i4_data.x));
        o4_data.y = 1.f / (1.f + expf(--i4_data.y));
        o4_data.z = 1.f / (1.f + expf(-i4_data.z));
        o4_data.w = 1.f / (1.f + expf(-i4_data.w));

        reinterpret_cast<float4 *>(o_data)[idx] = o4_data;
    }
}

