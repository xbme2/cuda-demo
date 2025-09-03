#include "cuda_header.h"
#include "softmax.cuh"
#include "sum.cuh"

void testAdd();
void testSoftmax();
int main() {
    testAdd();
    // testSoftmax();
    return 0;
}
void testSoftmax() {
    int N = 1 << 19, C = 1 << 5;
    float *i_data = (float *)malloc(N * C * sizeof(float));
    float *o_data = (float *)malloc(N * C * sizeof(float));
    InitByVal(i_data, N * C, 1.f);
    InitByVal(o_data, N * C, 0.f);
    CPU_TIME(softmax_cpu<float>(i_data, o_data, N, C));
}
void testAdd() {
    int N = 1 << 23;
    constexpr int BLOCK_SZ = 1 << 8;
    float *h = (float *)malloc(sizeof(float) * N);
    float *h_from_gpu = (float *)malloc(sizeof(float) * N);

    float *d = nullptr;
    float *d_out = nullptr;
    CHECK(cudaMalloc(&d, sizeof(float) * N));
    CHECK(cudaMalloc(&d_out, sizeof(float) * CEIL(N, BLOCK_SZ)));
    InitByVal(h, N, 1.f);
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));

    double current = currentTime();
    float sum = sum_cpu(h, N);
    std::cout << "cpu cost time: " << currentTime() - current << " ms" << "\t"
              << sum << std::endl;

    // v0

    KERNEL_TIME(1, sum_v0<BLOCK_SZ>
                <<<dim3(CEIL(N, BLOCK_SZ)), dim3(BLOCK_SZ)>>>(d, N));
    CHECK(cudaMemcpy(h_from_gpu, d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    float sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_from_gpu[i * BLOCK_SZ];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v0.5
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    KERNEL_TIME(1, sum_v0plus<BLOCK_SZ>
                <<<dim3(CEIL(N, BLOCK_SZ)), dim3(BLOCK_SZ)>>>(d, d_out, N));

    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v1
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    KERNEL_TIME(1, sum_v1<BLOCK_SZ>
                <<<dim3(CEIL(N, BLOCK_SZ)), dim3(BLOCK_SZ)>>>(d, d_out, N));
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v2
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    KERNEL_TIME(
        1, sum_v2<BLOCK_SZ>
        <<<dim3(CEIL(CEIL(N, BLOCK_SZ), 2)), dim3(BLOCK_SZ)>>>(d, d_out, N));
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), 2); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v3
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    KERNEL_TIME(
        1, sum_v3<BLOCK_SZ>
        <<<dim3(CEIL(CEIL(N, BLOCK_SZ), 2)), dim3(BLOCK_SZ)>>>(d, d_out, N));
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), 2); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v4
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    KERNEL_TIME(
        1, sum_v4<BLOCK_SZ>
        <<<dim3(CEIL(CEIL(N, BLOCK_SZ), 2)), dim3(BLOCK_SZ)>>>(d, d_out, N));
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), 2); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v5
    constexpr int NUM_PER_THREAD = 4;
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    KERNEL_TIME(
        1, sum_v5<BLOCK_SZ, NUM_PER_THREAD>
        <<<dim3(CEIL(CEIL(N, BLOCK_SZ), NUM_PER_THREAD)), dim3(BLOCK_SZ)>>>(
            d, d_out, N));
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), NUM_PER_THREAD); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v6
    // CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0, sizeof(float) * CEIL(N, BLOCK_SZ)));
    KERNEL_TIME(
        1, sum_v6<BLOCK_SZ, NUM_PER_THREAD>
        <<<dim3(CEIL(CEIL(N, BLOCK_SZ), NUM_PER_THREAD)), dim3(BLOCK_SZ)>>>(
            d, d_out, N));
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), NUM_PER_THREAD); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // free

    cudaFree(d);
    free(h);
    free(h_from_gpu);
}
