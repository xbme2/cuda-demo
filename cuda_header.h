#ifndef __CUDA_HEADER__
#define __CUDA_HEADER__

#include <format>
#include <iostream>
#include <stdio.h>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// 常量定义
constexpr int DEFAULT_BLOCK_NUM = 1 << 10;
constexpr int DEFAULT_BLOCK_SIZE = 1 << 8;

// 宏定义
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

#define VNAME(value) (#value)
#define CHECK_RESULT(src, dst, n) CheckResult(src, dst, n, #dst) // 记录变量名称
#define CPU_TIME(...)                                                          \
    cpuTime(#__VA_ARGS__, [&]() { __VA_ARGS__; }) // 记录函数名称
#define KERNEL_TIME(repeat, ...)                                               \
    kernelTime(repeat, #__VA_ARGS__, [&]() { __VA_ARGS__; }) // 记录函数名称

inline std::string clipName(const char *name) {
    std::string s(name);
    auto pos = s.find('<');
    if (pos != std::string::npos) {
        s = s.substr(0, pos);
    } else {
        if (s.size() > 20)
            s = s.substr(0, 20);
    }
    return s;
}

// 核函数，用于预热gpu
__global__ void warmup() {}

// 记时
// 用于cpu 记时
double currentTime() {
#ifdef _WIN32
    // windows
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)frequency.QuadPart * 1e3;
#else
    // linux/macos
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return (double)tp.tv_sec * 1e3 + (double)tp.tv_usec * 1e-3;
#endif
}

template <typename HostFunc> void cpuTime(const char *funcName, HostFunc f) {
    double current = currentTime();
    f();
    std::cout << clipName(funcName) << " cost time: " << currentTime() - current
              << " ms" << std::endl;
}

// 用于gpu 核函数记时
template <typename KernelFunc>
void kernelTime(int repeat, const char *funcName, KernelFunc kernel) {

    // 似乎最好是在测核函数之前warmup
    warmup<<<DEFAULT_BLOCK_NUM, DEFAULT_BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
    float totalTime = 0.f, singleTime = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeat; i++) {
        cudaEventRecord(start, 0);
        kernel();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&singleTime, start, stop);
        totalTime += singleTime;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << clipName(funcName) << " cost time: " << totalTime / repeat
              << " ms" << std::endl;
}

// 初始化
template <typename T> __host__ void InitByVal(T *data, int nElem, T val) {
    for (int i = 0; i < nElem; i++) {
        data[i] = val;
    }
}

template <typename T, typename Func>
__host__ void InitByFunc(T *data, int nElem, Func func) {
    for (int i = 0; i < nElem; i++) {
        data[i] = func(i);
    }
}

// 检测结果
template <typename T>
__host__ void CheckResult(T *src, T *dst, int nElem, const char *dstName) {
    // 检测变量名称
    if (strstr(dstName, "gpu") == nullptr) {
        std::cerr << "Error at line " << __LINE__ << " in file " << __FILE__
                  << ": dst variable (" << dstName
                  << ") must contain 'gpu' in its name!\n";

        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nElem; i++) {
        if (fabs(src[i] - dst[i]) > EPSILON) {
            printf("wrong happen at index %d", i);
            std::cout << ", want is " << src[i] << ", got is " << dst[i]
                      << std::endl;
            return;
        }
    }
    printf("check result success!\n");
}

#endif // __CUDA_HEADER__
