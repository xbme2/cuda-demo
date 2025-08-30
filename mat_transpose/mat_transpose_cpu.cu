#include "cuda_header.h"

__global__ void testTime(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid;
}

int main() {
    int n = 1 << 12;

    int blockSize = 1 << 10;

    dim3 block(blockSize);
    dim3 grid((n + blockSize - 1) / blockSize);

    float *h_data = (float *)malloc(sizeof(float) * n);
    memset(h_data, 0, sizeof(float) * n);
    float *d_data = nullptr;
    CHECK(cudaMalloc(&d_data, sizeof(float) * n));
    CHECK(
        cudaMemcpy(d_data, h_data, sizeof(float) * n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    testTime<<<grid, block>>>(d_data);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算时间差
    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms"
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    return 0;
}