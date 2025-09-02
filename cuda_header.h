#ifndef __CUDA_HEADER__
#define __CUDA_HEADER__

#include <format>
#include <iostream>
#include <stdio.h>
#define EPSILON 1e-3
#define CEIL(a, b) ((a + b - 1) / b)
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            printf("[ERROR]: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));   \
            exit(1);                                                           \
        }                                                                      \
    }

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// 用于cpu 记时
double currentTime() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return (double)tp.tv_sec * 1e3 + (double)tp.tv_usec * 1e-3;
}

template <typename T> void InitbyVal(T *data, int nElem, T val) {
    for (int i = 0; i < nElem; i++) {
        data[i] = val;
    }
}

template <typename T, typename Func>
void InitByFunc(T *data, int nElem, Func func) {
    for (int i = 0; i < nElem; i++) {
        data[i] = func(i);
    }
}

template <typename T> void CheckResult(T *src, T *dst, int nElem) {
    for (int i = 0; i < nElem; i++) {
        if (fabs(src[i] - dst[i]) > EPSILON) {
            printf("wrong happen at index %d", i);
            std::cout << ",\twant is" << src[i] << ", got is " << dst[i]
                      << std::endl;
            return;
        }
    }
    printf("check result success!\n");
}

__global__ void warmup() {}

template <typename kernelFunc, typename... Args>
float kernelTime(kernelFunc kernel, dim3 grid, dim3 block, int repeat,
                 Args... args) {

    // 似乎最好是在测核函数之前warmup
    warmup<<<grid, block>>>();
    cudaDeviceSynchronize();
    float totalTime = 0.f, singleTime = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeat; i++) {
        cudaEventRecord(start, 0);
        kernel<<<grid, block>>>(args...);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&singleTime, start, stop);
        totalTime += singleTime;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return totalTime / repeat;
}

#endif