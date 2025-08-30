#ifndef __CUDA_HEADER__
#define __CUDA_HEADER__

#include <format>
#include <iostream>
#include <stdio.h>
#define EPSILON 1e-6

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

double currentTime() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

template <typename T> void Init(T *data, int nElem, T val) {
    for (int i = 0; i < nElem; i++) {
        data[i] = val;
    }
}

template <typename T> void CheckResult(T *src, T *dst, int nElem) {
    for (int i = 0; i < nElem; i++) {
        if (fabs(src[i] - dst[i]) > EPSILON) {
            printf("wrong happen at index %d", i);
            std::cout << ",\twant is" << src[i] << ", got is " << dst[i]
                      << std::endl;
            break;
        }
    }
    printf("check result success!\n");
}

#endif