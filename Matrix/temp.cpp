#include <iostream>
#include <cstdlib>
#include <cstring>
#include "Header.h"


static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}


float* aligned_alloc32(int size) {
    void* ptr = std::aligned_alloc(32, sizeof(float) * size);
    std::memset(ptr, 0, sizeof(float) * size);
    return static_cast<float*>(ptr);
}

template<int n>
void matmul(const float* A_raw, const float* B_raw, float* C_raw) {
    // Hint the compiler that memory is 32-byte aligned
    const float* A = static_cast<const float*>(__builtin_assume_aligned(A_raw, 32));
    const float* B = static_cast<const float*>(__builtin_assume_aligned(B_raw, 32));
    float* C = static_cast<float*>(__builtin_assume_aligned(C_raw, 32));

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

float randomFloat(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());  // Seeded Mersenne Twister
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}
int main() {
  const int N = 1024;
  float* A = aligned_alloc32(N * N);
  float* B = aligned_alloc32(N * N);
  float* C = aligned_alloc32(N * N);
  for(int q = 0;q < N * N;q++) {
    A[q] = randomFloat(-1,1);
    B[q] = randomFloat(-1,1);
    C[q] = 0;
  }

  matmul<1024>(A,B,C);

  escape(C);
}