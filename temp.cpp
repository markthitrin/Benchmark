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
  for(int i = 8;i <= 1024;i*=4) {
    for(int j = 8;j <= 1024;j*=4) {
      for(int k = 8;k <= 1024;k*=4) {
        std::cout << "b->Args({" << i  << "," << j << "," << k << "});\n";
        if(k == 512) k/=2;
      }
      if(j == 512) j /=2;
    }
    if(i == 512) i /=2;
  }

  return 0;
}