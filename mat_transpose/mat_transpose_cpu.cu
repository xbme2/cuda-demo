#include "cuda_header.h"

constexpr int REPEAT = 5;
constexpr int MOD = 1000000;
constexpr int BLOCK_SZ = 32;
constexpr int NUM_PRE_THREAD = 4;

inline int Ceil(int a, int b) { return (a + b - 1) / b; }

// cpu 实现
void mat_transpose_cpu(float *i_data, float *o_data, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            o_data[j * M + i] = i_data[i * N + j];
        }
    }
}

// 上限
__global__ void mat_transpose_upLimit(float *i_data, float *o_data, int M,
                                      int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (iy < M && ix < N) {
        o_data[iy * N + ix] = i_data[iy * N + ix];
    }
}

// 下限
__global__ void mat_transpose_downLimit(float *i_data, float *o_data, int M,
                                        int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (iy < M && ix < N) {
        o_data[ix * M + iy] = i_data[ix * M + iy];
    }
}

// 读对齐
__global__ void mat_transpose_v0(float *i_data, float *o_data, int M, int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (iy < M && ix < N) {
        o_data[ix * M + iy] = i_data[iy * N + ix];
    }
}

// 写对齐
__global__ void mat_transpose_v00(float *i_data, float *o_data, int M, int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (iy < M && ix < N) {
        o_data[iy * N + ix] = i_data[ix * M + iy];
    }
}

// v1 使用共享内存优化
template <int BLOCK_SZ>
__global__ void mat_transpose_v1(float *i_data, float *o_data, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = bx * BLOCK_SZ + tx, iy = by * BLOCK_SZ + ty;

    __shared__ float temp[BLOCK_SZ][BLOCK_SZ];

    if (iy < M && ix < N) {
        temp[ty][tx] = i_data[iy * N + ix];
        // 顺序读，顺序写
    }
    __syncthreads();

    int nx = by * BLOCK_SZ + tx;
    int ny = bx * BLOCK_SZ + ty;
    if (ny < N && nx < M) {
        o_data[ny * M + nx] = temp[tx][ty];
        // 跨越读，顺序写
        // 导致一个warp 里 32 个线程访问同一列bank, 32路的bank conflict
    }
}

// v2 利用padding 解决bank conflict
template <int BLOCK_SZ>
__global__ void mat_transpose_v2(float *i_data, float *o_data, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = bx * BLOCK_SZ + tx, iy = by * BLOCK_SZ + ty;

    __shared__ float temp[BLOCK_SZ][BLOCK_SZ + 1]; //[32][33]

    if (iy < M && ix < N) {
        temp[ty][tx] = i_data[iy * N + ix];
        // 顺序读，顺序写
    }
    __syncthreads();

    int nx = by * BLOCK_SZ + tx;
    int ny = bx * BLOCK_SZ + ty;
    if (ny < N && nx < M) {
        o_data[ny * M + nx] = temp[tx][ty];
        // 跨越读，顺序写
    }
}

// v3 增加每个线程处理的数量，减少线程数量
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_v3(float *i_data, float *o_data, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = bx * BLOCK_SZ + tx, iy = by * BLOCK_SZ + ty;

    __shared__ float temp[BLOCK_SZ][BLOCK_SZ + 1]; //[32][33]

    int stride = BLOCK_SZ / NUM_PER_THREAD;
    if (ix < N) {
#pragma unroll
        for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
            if (iy + y_offset < M) {
                temp[ty + y_offset][tx] = i_data[(iy + y_offset) * N + ix];
            }
        }
    }
    __syncthreads();

    int nx = by * BLOCK_SZ + tx;
    int ny = bx * BLOCK_SZ + ty;
    if (nx < M) {
#pragma unroll
        for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
            if (ny + y_offset < N)
                o_data[(ny + y_offset) * M + nx] = temp[tx][ty + y_offset];
        }
    }
}

// v4 减少分支判断
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_v4(float *i_data, float *o_data, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = bx * BLOCK_SZ + tx, iy = by * BLOCK_SZ + ty;

    __shared__ float temp[BLOCK_SZ][BLOCK_SZ + 1]; //[32][33]

    int stride = BLOCK_SZ / NUM_PER_THREAD;
    if (ix < N) {
        if (iy + BLOCK_SZ <= M) {
#pragma unroll
            for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
                temp[ty + y_offset][tx] = i_data[(iy + y_offset) * N + ix];
            }
        } else {
            for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
                if (iy + y_offset < M) {
                    temp[ty + y_offset][tx] = i_data[(iy + y_offset) * N + ix];
                }
            }
        }
    }
    __syncthreads();

    int nx = by * BLOCK_SZ + tx;
    int ny = bx * BLOCK_SZ + ty;
    if (nx < M) {
        if (ny + BLOCK_SZ <= N) {
#pragma unroll
            for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
                o_data[(ny + y_offset) * M + nx] = temp[tx][ty + y_offset];
            }
        } else {
            for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
                if (ny + y_offset < N)
                    o_data[(ny + y_offset) * M + nx] = temp[tx][ty + y_offset];
            }
        }
    }
}

// v5 修改NUM_PER_THREAD
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_v5(float *i_data, float *o_data, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = bx * BLOCK_SZ + tx, iy = by * BLOCK_SZ + ty;

    __shared__ float temp[BLOCK_SZ][BLOCK_SZ + 1]; //[32][33]

    int stride = BLOCK_SZ / NUM_PER_THREAD;
    if (ix < N) {
#pragma unroll
        for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
            if (iy + y_offset < M) {
                temp[ty + y_offset][tx] = i_data[(iy + y_offset) * N + ix];
            }
        }
    }
    __syncthreads();

    int nx = by * BLOCK_SZ + tx;
    int ny = bx * BLOCK_SZ + ty;
    if (nx < M) {
#pragma unroll
        for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
            if (ny + y_offset < N)
                o_data[(ny + y_offset) * M + nx] = temp[tx][ty + y_offset];
        }
    }
}

// v6 __restrict__
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_v6(float *__restrict__ i_data,
                                 float *__restrict__ o_data, int M, int N) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = bx * BLOCK_SZ + tx, iy = by * BLOCK_SZ + ty;

    __shared__ float temp[BLOCK_SZ][BLOCK_SZ + 1]; //[32][33]

    int stride = BLOCK_SZ / NUM_PER_THREAD;
    if (ix < N) {
#pragma unroll
        for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
            if (iy + y_offset < M) {
                temp[ty + y_offset][tx] = i_data[(iy + y_offset) * N + ix];
            }
        }
    }
    __syncthreads();

    int nx = by * BLOCK_SZ + tx;
    int ny = bx * BLOCK_SZ + ty;
    if (nx < M) {
#pragma unroll
        for (int y_offset = 0; y_offset < BLOCK_SZ; y_offset += stride) {
            if (ny + y_offset < N)
                o_data[(ny + y_offset) * M + nx] = temp[tx][ty + y_offset];
        }
    }
}

int main() {
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
    float tmin = kernelTime(mat_transpose_upLimit, grid, block, REPEAT, d_idata,
                            d_odata, M, N);
    std::cout << "kernel minest time: " << tmin << " ms" << std::endl;
    // CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
    //                  cudaMemcpyDeviceToHost));
    // CheckResult(h_data, h_data_from_gpu, M * N);
    // 下限,跨越读，跨越写
    float tmax = kernelTime(mat_transpose_downLimit, grid, block, REPEAT,
                            d_idata, d_odata, M, N);
    std::cout << "kernel maxest time: " << tmax << " ms" << std::endl;

    // mat_transpose_v0
    float t0 = kernelTime(mat_transpose_v0, grid, block, REPEAT, d_idata,
                          d_odata, M, N);
    std::cout << "v0 kernel avg time: " << t0 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);
    // mat_transpose_v00
    float t1 = kernelTime(mat_transpose_v00, grid, block, REPEAT, d_idata,
                          d_odata, M, N);
    std::cout << "v00 kernel avg time: " << t1 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v1
    float t2 = kernelTime(mat_transpose_v1<BLOCK_SZ>, grid, block, REPEAT,
                          d_idata, d_odata, M, N);
    std::cout << "v1 kernel avg time: " << t2 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v2
    float t3 = kernelTime(mat_transpose_v2<BLOCK_SZ>, grid, block, REPEAT,
                          d_idata, d_odata, M, N);
    std::cout << "v2 kernel avg time: " << t3 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v3
    block = dim3(BLOCK_SZ, BLOCK_SZ / NUM_PRE_THREAD);
    grid = dim3(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    float t4 = kernelTime(mat_transpose_v3<BLOCK_SZ, NUM_PRE_THREAD>, grid,
                          block, REPEAT, d_idata, d_odata, M, N);
    std::cout << "v3 kernel avg time: " << t4 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);
    // std::cout << h_data_from_gpu[1] << std::endl;

    // mat_transpose_v4
    float t5 = kernelTime(mat_transpose_v4<BLOCK_SZ, NUM_PRE_THREAD>, grid,
                          block, REPEAT, d_idata, d_odata, M, N);
    std::cout << "v4 kernel avg time: " << t5 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v5
    block = dim3(BLOCK_SZ, BLOCK_SZ / 8);
    grid = dim3(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    float t6 = kernelTime(mat_transpose_v5<BLOCK_SZ, 8>, grid, block, REPEAT,
                          d_idata, d_odata, M, N);
    std::cout << "v5 kernel avg time: " << t6 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);

    // mat_transpose_v6
    block = dim3(BLOCK_SZ, BLOCK_SZ / NUM_PRE_THREAD);
    grid = dim3(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    float t7 = kernelTime(mat_transpose_v6<BLOCK_SZ, NUM_PRE_THREAD>, grid,
                          block, REPEAT, d_idata, d_odata, M, N);
    std::cout << "v6 kernel avg time: " << t7 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_data_from_gpu, d_odata, sizeof(float) * M * N,
                     cudaMemcpyDeviceToHost));
    CheckResult(h_data, h_data_from_gpu, M * N);

    // free

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_data);
    free(h_data_from_gpu);
    return 0;
}