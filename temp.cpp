#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <random>
#include <iostream>
#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)


int main() {
  void* memory;
  const int array_size = 2048 / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  volatile __m256i* const p0 = static_cast<__m256i*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(__m256i);
  __m256i sink0; memset(&sink0, 0x1b, sizeof(sink0));
  __m256i sink = sink0;

  int* ind0 = (int*)malloc(sizeof(int) * array_size);
  for(int i = 0;i < array_size;i++) {
    ind0[i] = i;
  }
  std::shuffle(ind0, ind0 + array_size, std::default_random_engine(0));
  volatile int u = 0;

  __asm volatile("# LLVM-MCA-BEGIN":::"memory");
  volatile __m256i* p = p0;
  const int* ind = ind0;
  while(ind != ind0 + array_size) {
    REPEAT(sink = *(p0 + *ind++););
  }
  __asm volatile("# LLVM-MCA-END":::"memory");

  free(memory);

  return 0;
}