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

int main() {
    int N = 1 << 14;
    int blockSize = 1 << 8;

    float *h_a = (float *)malloc(sizeof(float) * N);
    float *h_b = (float *)malloc(sizeof(float) * N);
    float *h_c = (float *)malloc(sizeof(float) * N);
    float *h_c_from_gpu = (float *)malloc(sizeof(float) * N);
    InitByVal(h_a, N, 1.f);
    InitByVal(h_b, N, 1.f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK(cudaMalloc(&d_a, sizeof(float) * N));
    CHECK(cudaMalloc(&d_b, sizeof(float) * N));
    CHECK(cudaMalloc(&d_c, sizeof(float) * N));
    CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c, h_c, sizeof(float) * N, cudaMemcpyHostToDevice));

    double current = currentTime();
    add_cpu(h_a, h_b, h_c, N);
    std::cout << "cpu cost time: " << currentTime() - current << " ms"
              << std::endl;

    dim3 block(blockSize);
    dim3 grid(CEIL(N, blockSize));

    float t0 = kernelTime(add_v0, grid, block, 3, d_a, d_b, d_c, N);
    std::cout << "kernel v0 cost time: " << t0 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_c_from_gpu, d_c, sizeof(float) * N,
                     cudaMemcpyDeviceToHost));

    CHECK_RESULT(h_c_from_gpu, h_c, N);

    float t1 = kernelTime(add_vf, CEIL(CEIL(N, 4), blockSize), block, 3, d_a,
                          d_b, d_c, N);
    std::cout << "kernel vf cost time: " << t1 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_c_from_gpu, d_c, sizeof(float) * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_c_from_gpu, h_c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_from_gpu);
}