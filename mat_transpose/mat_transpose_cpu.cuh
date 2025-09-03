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

