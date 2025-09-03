#include "cuda_header.h"
#include "mat_transpose_cpu.cuh"

void testMatTranspose();
int main() {
    testMatTranspose();
    return 0;
}
void testMatTranspose() {
    int M = 1 << 12, N = 1 << 12;

    int blockSize = 32;

    dim3 block(blockSize,
               blockSize); // grid 和 block的中维度设置与多维数组中的表示相反,
    dim3 grid(N / blockSize, M / blockSize);

    float *h_data = (float *)malloc(sizeof(float) * M * N);
    float *i_data = (float *)malloc(sizeof(float) * M * N);
    float *h_data_from_gpu = (float *)malloc(sizeof(float) * M * N);
    InitByFunc(i_data, M * N, [](int i) { return i % MOD; });

    float *d_idata = nullptr, *d_odata = nullptr;
    CHECK(cudaMalloc(&d_idata, sizeof(float) * M * N));
    CHECK(cudaMalloc(&d_odata, sizeof(float) * M * N));

    CHECK(cudaMemcpy(d_idata, i_data, sizeof(float) * M * N,
                     cudaMemcpyHostToDevice));
    double current = currentTime();
    mat_transpose_cpu(i_data, h_data, M, N);
    std::cout << "cpu cost time: " << currentTime() - current << " ms"
              << std::endl;

    warmup<<<grid, block>>>();
    cudaDeviceSynchronize();
    // 存疑,是在每次测量时间前warmup，还是整体的之前warmup

    // 上限,自我复制
    KERNEL_TIME(REPEAT,
                mat_transpose_upLimit<<<grid, block>>>(d_idata, d_odata, M, N));
    // CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
    //                  cudaMemcpyDeviceToHost));
    // CHECK_RESULT(h_data, h_data_from_gpu, M * N);
    // 下限,跨越读，跨越写
    KERNEL_TIME(REPEAT, mat_transpose_downLimit<<<grid, block>>>(
                            d_idata, d_odata, M, N));

    // mat_transpose_v0
    KERNEL_TIME(REPEAT,
                mat_transpose_v0<<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);
    // mat_transpose_v00
    KERNEL_TIME(REPEAT,
                mat_transpose_v00<<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v1
    KERNEL_TIME(REPEAT, mat_transpose_v1<BLOCK_SZ>
                <<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v2
    KERNEL_TIME(REPEAT, mat_transpose_v2<BLOCK_SZ>
                <<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v3
    block = dim3(BLOCK_SZ, BLOCK_SZ / NUM_PRE_THREAD);
    grid = dim3(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    KERNEL_TIME(REPEAT, mat_transpose_v3<BLOCK_SZ, NUM_PRE_THREAD>
                <<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);
    // std::cout << h_data_from_gpu[1] << std::endl;

    // mat_transpose_v4
    KERNEL_TIME(REPEAT, mat_transpose_v4<BLOCK_SZ, NUM_PRE_THREAD>
                <<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v5
    block = dim3(BLOCK_SZ, BLOCK_SZ / 8);
    grid = dim3(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    KERNEL_TIME(REPEAT, mat_transpose_v5<BLOCK_SZ, 8>
                <<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v6
    block = dim3(BLOCK_SZ, BLOCK_SZ / NUM_PRE_THREAD);
    grid = dim3(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    KERNEL_TIME(REPEAT, mat_transpose_v6<BLOCK_SZ, NUM_PRE_THREAD>
                <<<grid, block>>>(d_idata, d_odata, M, N));
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CHECK_RESULT(h_data, h_data_from_gpu, M * N);

    // free

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_data);
    free(h_data_from_gpu);
}
