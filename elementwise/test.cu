#include "cuda_header.h"
#include "elementwise.cuh"

void testElemensive();
int main() {
    testElemensive();
    return 0;
}
void testElemensive() {
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

    KERNEL_TIME(3, add_v0<<<grid, block>>>(d_a, d_b, d_c, N));
    CHECK(cudaMemcpy(h_c_from_gpu, d_c, sizeof(float) * N,
                     cudaMemcpyDeviceToHost));

    CHECK_RESULT(h_c_from_gpu, h_c, N);

    KERNEL_TIME(
        3, add_vf<<<CEIL(CEIL(N, 4), blockSize), block>>>(d_a, d_b, d_c, N));
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
