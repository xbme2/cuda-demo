#include "cuda_header.h"
#define WARP_SIZE 32
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

template <int BLOCK_SZ>
__global__ void sum_v0plus(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    constexpr int PAD = 0; // BLOCK_SZ / 32 BLOCK_SZ >> 5
    __shared__ float smem[BLOCK_SZ + PAD];
    // float *start = &i_data[];
    smem[tx] = i_data[bx * BLOCK_SZ + tx];
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < BLOCK_SZ; stride <<= 1) {
        if (tx % (2 * stride) == 0) {
            smem[tx] += smem[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0)
        o_data[bx] = smem[0];
}

// 显然涉及对全局内存多次读取使用共享内存优化
template <int BLOCK_SZ>
__global__ void sum_v1(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    constexpr int PAD = 0; // BLOCK_SZ / 32 BLOCK_SZ >> 5
    __shared__ float smem[BLOCK_SZ + PAD];
    // float *start = &i_data[];
    smem[tx] = i_data[bx * BLOCK_SZ + tx];
    __syncthreads();
#pragma unroll
    for (int stride = BLOCK_SZ >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            smem[tx] += smem[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0)
        o_data[bx] = smem[0];
}

template <int BLOCK_SZ>
__global__ void sum_v2(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    constexpr int PAD = 0; // BLOCK_SZ / 32 BLOCK_SZ >> 5
    __shared__ float smem[BLOCK_SZ + PAD];
    // float *start = &i_data[];
    smem[tx] = i_data[bx * BLOCK_SZ * 2 + tx] +
               i_data[bx * BLOCK_SZ * 2 + tx + BLOCK_SZ];
    __syncthreads();
#pragma unroll
    for (int stride = BLOCK_SZ >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            smem[tx] += smem[tx + stride];
        }
        __syncthreads();
        // 当s小于 32时,只有一个warp工作，没必要同步
    }
    if (tx == 0)
        o_data[bx] = smem[0];
}

__device__ void warpReduce(volatile float *smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

template <typename T> __device__ T warpReduce_v1(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SZ>
__global__ void sum_v3(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    constexpr int PAD = 0; // BLOCK_SZ / 32 BLOCK_SZ >> 5
    __shared__ float smem[BLOCK_SZ + PAD];
    // float *start = &i_data[];
    smem[tx] = i_data[bx * BLOCK_SZ * 2 + tx] +
               i_data[bx * BLOCK_SZ * 2 + tx + BLOCK_SZ];
    __syncthreads();
#pragma unroll
    for (int stride = BLOCK_SZ >> 1; stride > 32; stride >>= 1) {
        if (tx < stride) {
            smem[tx] += smem[tx + stride];
        }
        __syncthreads();
    }

    if (tx < 32)
        warpReduce(smem, tx);
    if (tx == 0)
        o_data[bx] = smem[0];
}

template <int BLOCK_SZ>
__global__ void sum_v4(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    constexpr int PAD = 0; // BLOCK_SZ / 32 BLOCK_SZ >> 5
    __shared__ float smem[BLOCK_SZ + PAD];
    // float *start = &i_data[];
    smem[tx] = i_data[bx * BLOCK_SZ * 2 + tx] +
               i_data[bx * BLOCK_SZ * 2 + tx + BLOCK_SZ];
    __syncthreads();
    if (BLOCK_SZ >= 512) {
        if (tx < 256) {
            smem[tx] += smem[tx + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SZ >= 256) {
        if (tx < 128) {
            smem[tx] += smem[tx + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SZ >= 128) {
        if (tx < 64) {
            smem[tx] += smem[tx + 64];
        }
        __syncthreads();
    }

    if (tx < 32)
        warpReduce(smem, tx);
    if (tx == 0)
        o_data[bx] = smem[0];
}

template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void sum_v5(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    constexpr int PAD = 0; // BLOCK_SZ / 32 BLOCK_SZ >> 5
    __shared__ float smem[BLOCK_SZ + PAD];
    smem[tx] = 0.f; // 初始值未定义,可能是之前运行的 kernel/block
                    // 残留的数据,或者干脆就是随机垃圾值
// float *start = &i_data[];
#pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        smem[tx] += i_data[bx * BLOCK_SZ * NUM_PER_THREAD + tx + BLOCK_SZ * i];
    }
    __syncthreads();
    if (BLOCK_SZ >= 512) {
        if (tx < 256) {
            smem[tx] += smem[tx + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SZ >= 256) {
        if (tx < 128) {
            smem[tx] += smem[tx + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SZ >= 128) {
        if (tx < 64) {
            smem[tx] += smem[tx + 64];
        }
        __syncthreads();
    }

    if (tx < 32)
        warpReduce(smem, tx);
    if (tx == 0)
        o_data[bx] = smem[0];
}

template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void sum_v6(float *i_data, float *o_data, int N) {
    int bx = blockIdx.x, tx = threadIdx.x;
    int laneId = tx % WARP_SIZE;
    int warpId = tx / WARP_SIZE;

    float val = 0.f;
#pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        val += i_data[bx * BLOCK_SZ * NUM_PER_THREAD + tx + BLOCK_SZ * i];
    }

    val = warpReduce_v1(val);
    // __syncthreads();
    if (BLOCK_SZ < 128) {
        if (laneId == 0) {
            atomicAdd(&o_data[bx], val);
        }
    } else {
        __shared__ float smem[WARP_SIZE]; // 由于tx < 1024 ，所以tx / 32 < 32;
        if (laneId == 0) {
            smem[warpId] = val;
        }
        __syncthreads();
        if (warpId == 0) {
            val = (tx < CEIL(BLOCK_SZ, WARP_SIZE)) ? smem[tx] : 0.f;
            val = warpReduce_v1(val);
        }

        if (tx == 0)
            o_data[bx] = val;
    }
}

int main() {
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
    float t0 = kernelTime(sum_v0<BLOCK_SZ>, dim3(CEIL(N, BLOCK_SZ)),
                          dim3(BLOCK_SZ), 1, d, N);
    std::cout << "native kernel cost time: " << t0 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_from_gpu, d, sizeof(float) * N, cudaMemcpyDeviceToHost));
    float sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_from_gpu[i * BLOCK_SZ];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v0.5
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    float t0plus = kernelTime(sum_v0plus<BLOCK_SZ>, dim3(CEIL(N, BLOCK_SZ)),
                              dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "t0plus share memory kernel cost time: " << t0plus << " ms"
              << std::endl;
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v1
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    float t1 = kernelTime(sum_v1<BLOCK_SZ>, dim3(CEIL(N, BLOCK_SZ)),
                          dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "share memory kernel cost time: " << t1 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(N, BLOCK_SZ); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v2
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    float t2 = kernelTime(sum_v2<BLOCK_SZ>, dim3(CEIL(CEIL(N, BLOCK_SZ), 2)),
                          dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "v2  kernel cost time: " << t2 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), 2); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v3
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    float t3 = kernelTime(sum_v3<BLOCK_SZ>, dim3(CEIL(CEIL(N, BLOCK_SZ), 2)),
                          dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "v3  kernel cost time: " << t3 << " ms" << std::endl;
    CHECK(cudaMemcpy(h_from_gpu, d_out, sizeof(float) * CEIL(N, BLOCK_SZ),
                     cudaMemcpyDeviceToHost));
    sum_gpu = 0.f;
    for (int i = 0; i < CEIL(CEIL(N, BLOCK_SZ), 2); i++) {
        sum_gpu += h_from_gpu[i];
    }
    CHECK_RESULT(&sum, &sum_gpu, 1);

    // v4
    CHECK(cudaMemcpy(d, h, sizeof(float) * N, cudaMemcpyHostToDevice));
    float t4 = kernelTime(sum_v4<BLOCK_SZ>, dim3(CEIL(CEIL(N, BLOCK_SZ), 2)),
                          dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "v4  kernel cost time: " << t4 << " ms" << std::endl;
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
    float t5 = kernelTime(sum_v5<BLOCK_SZ, NUM_PER_THREAD>,
                          dim3(CEIL(CEIL(N, BLOCK_SZ), NUM_PER_THREAD)),
                          dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "v5  kernel cost time: " << t5 << " ms" << std::endl;
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
    float t6 = kernelTime(sum_v6<BLOCK_SZ, NUM_PER_THREAD>,
                          dim3(CEIL(CEIL(N, BLOCK_SZ), NUM_PER_THREAD)),
                          dim3(BLOCK_SZ), 1, d, d_out, N);
    std::cout << "v6  kernel cost time: " << t6 << " ms" << std::endl;
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

    return 0;
}