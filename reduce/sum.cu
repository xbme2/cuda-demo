#include "cuda_header.h"

float sum_cpu(float *i_data, int N) {
    float sum = 0.f;
    for (int i = 0; i < N; i++) {
        sum += i_data[i];
    }
    return sum;
}

template <int BLOCK_SZ> __global__ void sum_v0(float *i_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    float *start = &i_data[bx * BLOCK_SZ];
#pragma unroll
    for (int stride = BLOCK_SZ >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tx < stride) {
            start[tx] += start[tx + stride];
        }
    }
}

// 显然涉及对全局内存多次读取使用共享内存优化,并且加上padding
// template <int BLOCK_SZ> __global__ void sum_v2(float *i_data, int N) {
//     int bx = blockIdx.x, tx = threadIdx.x;
//     constexpr int PAD = BLOCK_SZ >> 5; // BLOCK_SZ / 32
//     __shared__ float smem[BLOCK_SZ + PAD];
//     float *start = &i_data[bx * BLOCK_SZ];
//     __shared__ float temp[32 * 33];
//     temp[tx] = start[tx];
// #pragma unroll
//     for (int stride = BLOCK_SZ >> 1; stride > 0; stride >>= 1) {
//         __syncthreads();
//         if (tx < stride) {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     if (tx == 0)
//         start[0] = temp[0];
// }

int main() {
    int N = 1 << 18;
    constexpr int BLOCK_SZ = 1 << 10;
    float *h = (float *)malloc(sizeof(float) * N);
    float *h_grom_gpu = (float *)malloc(sizeof(float) * N);

    float *d = nullptr;
    CHECK(cudaMalloc(&d, sizeof(float) * N));
    InitByVal(h, N, 1.f);
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));

    double current = currentTime();
    float sum = sum_cpu(h, N);
    std::cout << "cpu cost time: " << currentTime() - current << " ms" << "\t"
              << sum << std::endl;

    // v0
    float t0 = kernelTime(sum_v0<BLOCK_SZ>, dim3(CEIL(N, BLOCK_SZ)),
                          dim3(BLOCK_SZ), 1, d, N);
    std::cout << "native kernel cost time: " << t0 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_grom_gpu, d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    float sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_grom_gpu[i * BLOCK_SZ];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v1
    float t1 = kernelTime(sum_v0<BLOCK_SZ>, dim3(CEIL(N, BLOCK_SZ)),
                          dim3(BLOCK_SZ), 1, d, N);
    std::cout << "share memory kernel cost time: " << t1 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_grom_gpu, d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_grom_gpu[i * BLOCK_SZ];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    cudaFree(d);
    free(h);
    free(h_grom_gpu);

    return 0;
}