#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Detected CUDA Devices: " << deviceCount << "\n\n";

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "=== Device " << device << " ===\n";
        std::cout << "Name: " << prop.name << "\n";
        std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
        std::cout << "Shared memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB\n";
        std::cout << "Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << "\n";
        std::cout << "Warp size: " << prop.warpSize << "\n";
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Max block dimensions: [" << prop.maxThreadsDim[0]
                  << ", " << prop.maxThreadsDim[1]
                  << ", " << prop.maxThreadsDim[2] << "]\n";
        std::cout << "Max grid dimensions: [" << prop.maxGridSize[0]
                  << ", " << prop.maxGridSize[1]
                  << ", " << prop.maxGridSize[2] << "]\n";
        std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
        std::cout << "Clock rate: " << prop.clockRate / 1000 << " MHz\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "\n";
    }

    return 0;
}