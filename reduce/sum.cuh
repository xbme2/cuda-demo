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
