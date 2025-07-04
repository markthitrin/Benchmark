#include "Header.cuh"
#include "Tensor.cuh"


int main() {
    Tensor a(10,10);
    Tensor b(10,10);
    Tensor c(10,10);
    float* in1 = nullptr;
    float* in2 = nullptr;
    cudaMallocHost(&in1, 10 * 10 * sizeof(float));
    cudaMallocHost(&in2, 10 * 10 * sizeof(float));
    for(int q = 0;q < 100;q++) {
        in1[q] = std::rand();
        in2[q] = -in1[q];
    }
    fromArray(in1,a);
    fromArray(in2,b);
    cudaDeviceSynchronize();

    plusAsync(a,b,c);
    cudaDeviceSynchronize();

    float* out = nullptr;
    cudaMallocHost(&out, 10 * 10 * sizeof(float));
    cudaDeviceSynchronize();

    toArray(c,out);
    cudaDeviceSynchronize();

    for(int q = 0;q < 10;q++) {
        for(int w = 0;w < 10;w++) {
            std::cout << out[q * 10 + w] << " ";
        }
        std::cout << std::endl;
    }
}