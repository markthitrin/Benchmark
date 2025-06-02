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
  const int z = 16 * 1024;
  int a,b,c;
  float bestCost = 10000;
  for(int q = 0;q < 512;q++) {
    for(int w = 0;w < 512;w++) {
      for(int e = 0;e < 512;e++) {
        float cost = 1.0f / q + 1.0f/w + 1.0f/e;
        if(cost < bestCost && q *w + w * e + e * q < z) {
          a = q;
          b = w;
          c = e;
          bestCost = cost;
        }
      }
    }
  }
  std::cout << bestCost << std::endl;
  std::cout << a << " " << b << " " << c << std::endl;
  return 0;
}