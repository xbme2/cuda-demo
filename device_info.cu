#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Total global memory: " << (prop.totalGlobalMem >> 20) << " MB"
              << std::endl;
    std::cout << "Shared memory per block: " << (prop.sharedMemPerBlock >> 10)
              << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits"
              << std::endl;
    std::cout << "Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz"
              << std::endl;
    return 0;
}