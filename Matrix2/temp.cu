#include <cuda_runtime.h>
#include <iostream>
#include "Tensor.cuh"
#include "Header.cuh"

void Print(Tensor A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
	A.toFloat(_A);

    for(int i = 0;i < r;i++) {
        for(int j = 0;j < c;j++) {
            std::cout << _A[i * A.col + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Print2(Tensor A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
	A.toFloat(_A);

    std::cout << "{";
    for(int i = 0;i < r;i++) {
        std::cout << "{";
        for(int j = 0;j < c;j++) {
            std::cout << _A[i * A.col + j];
            if(j + 1 != c) std::cout << ",";
        }std::cout << "}";
        if(i + 1 != r) std::cout << ",";
        
    }
    std::cout << "}\n";
}


void Print2T(Tensor A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
	A.toFloat(_A);

    std::cout << "{";
    for(int i = 0;i < c;i++) {
        std::cout << "{";
        for(int j = 0;j < r;j++) {
            std::cout << _A[j * A.col + i];
            if(j + 1 != r) std::cout << ",";
        }std::cout << "}";
        if(i + 1 != c) std::cout << ",";
        
    }
    std::cout << "}\n";
}

float randomFloat(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());  // Seeded Mersenne Twister
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}


void randTensor(Tensor& a) {
  float* in = new float[a.row * a.col];
  for(int q = 0;q < a.row * a.col;q++) {
    in[q] = randomFloat(-1,1);
  }
  fromArray(in,a);
}


// Based on NVIDIA CUDA programming guide: CUDA cores per SM by architecture
int getCudaCoresPerSM(cudaDeviceProp& prop) {
    int coresPerSM = 0;
    switch (prop.major) {
        case 8: // Ampere
            coresPerSM = (prop.minor == 0) ? 64 : 128;
            break;
        case 7: // Volta/Turing
            coresPerSM = (prop.minor == 5) ? 64 : 64;
            break;
        case 6: // Pascal
            coresPerSM = (prop.minor == 1 || prop.minor == 2) ? 128 : 64;
            break;
        case 5: // Maxwell
        case 3: // Kepler
            coresPerSM = 128;
            break;
        case 2: // Fermi
            coresPerSM = (prop.minor == 1) ? 48 : 32;
            break;
        default:
            std::cerr << "Unknown architecture, using default 64 cores/SM\n";
            coresPerSM = 64;
    }
    return coresPerSM;
}

void printSpec() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int coresPerSM = getCudaCoresPerSM(prop);
    int totalCores = prop.multiProcessorCount * coresPerSM;

    float clockGHz = prop.clockRate * 1e-6f;  // from kHz to GHz
    float memoryClockGHz = prop.memoryClockRate * 1e-6f;  // from kHz to GHz
    int memBusWidth = prop.memoryBusWidth;  // in bits

    float tflops = totalCores * clockGHz * 2 / 1000.0f;  // 2 FLOPs per cycle
    float memBandwidth = 2.0f * memoryClockGHz * (memBusWidth / 8.0f);  // in GB/s

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "CUDA cores: " << totalCores << "\n";
    std::cout << "Core clock: " << clockGHz << " GHz\n";
    std::cout << "Memory clock: " << memoryClockGHz << " GHz\n";
    std::cout << "Memory bus width: " << memBusWidth << " bits\n";
    std::cout << "Theoretical FP32 Peak: " << tflops << " TFLOPS\n";
    std::cout << "Theoretical Memory Bandwidth: " << memBandwidth << " GB/s\n";
}

int main() {
    Tensor a(17,1);
    Tensor b(17,1);
    Tensor c(1,1);
    randTensor(a);
    randTensor(b);
    randTensor(c);
    cudaDeviceSynchronize();
    Print2T(a,0,0,17,1);
    Print2(b,0,0,17,1);
    
    MatMulPlusAsync(a,b,c,true,false);
    Print(c,0,0,1,1);
    cudaDeviceSynchronize();
    

    return 0;
}