#include <benchmark/benchmark.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <numeric>
#include <string>
#include <random>
#include <cmath>
#include <set>
#include <map>
#include "Header.h"

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

std::random_device rd;
std::mt19937 gen(rd());

int getRandomInt(int min, int max) {
  std::uniform_int_distribution<> dist(min, max);
  return dist(gen);
}

float randomFloatFromBits() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

  uint32_t bits = dist(gen);

  float result;
  std::memcpy(&result, &bits, sizeof(float));

  return result;
}

float randomFloat(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());  // Seeded Mersenne Twister
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}

bool check(const int size, float* a, float* b) {
  for(int q = 0;q < size;q++) {
    if(a[q] != b[q] && !(std::isnan(a[q]) && std::isnan(b[q]))) {
      std::cout << a[q] << " : " << b[q] << std::endl;
      return false;
    }
  }
  return true;
}

bool check(const int size, const Matrix& a, float* b) {
  for(int q = 0;q < size;q++) {
    if(a.data[q] != b[q] && !(std::isnan(a.data[q]) && std::isnan(b[q]))) {
      std::cout << a.data[q] << " : " << b[q] << std::endl;
      return false;
    }
  }
  return true;
}

static void MatrixMul1(benchmark::State& state) { // call operator*
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d1, d2);
  Matrix m(d2, d3);
  Matrix out(d1, d3);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    out = t * m;
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatrixMul2(benchmark::State& state) {       // implemented ymm
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d1, d2);
  Matrix m(d2, d3);
  Matrix out(d1, d3);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d2;q++) {
      for(int w = 0;w < d1;w++) {
        for(int e = 0;e < d3;e++) {
          out.data.data()[w * d3 + e] += t.data.data()[w * d2 + q] * m.data.data()[q * d3 + e];
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
static void MatrixMul3(benchmark::State& state) {   //implemented using xmm
  // L1_CACHE 256 * 1024
  // L1 instance 32 * 1024 
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d1, d2);
  Matrix m(d2, d3);
  Matrix out(d1, d3);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d1;q++) {
      for(int e = 0;e < d3;e++) {
        for(int w = 0;w < d2;w++) {
          out.data.data()[q * d3 + e] += t.data.data()[q * d2 + w] * m.data.data()[w * d3 + e];     // do this to prevent using operator[]
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatrixMul4(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d1, d2);
  Matrix m(d3, d2);
  Matrix out(d1, d3);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      m[w][q] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d1;q++) {
      for(int w = 0;w < d3;w++) {
        for(int e = 0;e < d2;e++) {
          out.data.data()[q * d3 + w] = t.data.data()[q * d2 + e] * m.data.data()[w * d2 + e];
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatrixMul5(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d2, d1);
  Matrix m(d2, d3);
  Matrix out(d1, d3);
  for(int w = 0;w < d2;w++) {
    for(int e = 0;e < d1;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int w = 0;w < d2;w++) {
      for(int q = 0;q < d1;q++) {
        for(int e = 0;e < d3;e++) {
          out.data.data()[q * d3 + e] = t.data.data()[w * d1 + q] * m.data.data()[w * d3 + e];
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatrixMul6(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d2, d1);
  Matrix m(d2, d3);
  Matrix out(d3, d1);
  for(int w = 0;w < d2;w++) {
    for(int e = 0;e < d1;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int w = 0;w < d2;w++) {
      for(int q = 0;q < d1;q++) {
        for(int e = 0;e < d3;e++) {
          out.data.data()[e * d1 + q] = t.data.data()[w * d1 + q] * m.data.data()[w * d3 + e];
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatrixMul7(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d1, d2);
  Matrix m(d3, d2);
  Matrix out(d1, d3);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d3;q++) {
    for(int w = 0;w < d2;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int w = 0;w < d2;w++) {
      for(int q = 0;q < d1;q++) {
        for(int e = 0;e < d3;e++) {
          out.data.data()[q * d3 + e] = t.data.data()[w * d1 + q] * m.data.data()[w * d3 + e];
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatrixMul8(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Matrix t(d1, d2);
  Matrix m(d3, d2);
  Matrix out(d3, d1);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d3;q++) {
    for(int w = 0;w < d2;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int w = 0;w < d2;w++) {
      for(int q = 0;q < d1;q++) {
        for(int e = 0;e < d3;e++) {
          out.data.data()[e * d1 + q] = t.data.data()[w * d1 + q] * m.data.data()[w * d3 + e];
        }
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  b->Args({8,8,8});
  b->Args({8,8,32});
  b->Args({8,8,128});
  b->Args({8,8,512});
  b->Args({8,8,1024});
  b->Args({8,32,8});
  b->Args({8,32,32});
  b->Args({8,32,128});
  b->Args({8,32,512});
  b->Args({8,32,1024});
  b->Args({8,128,8});
  b->Args({8,128,32});
  b->Args({8,128,128});
  b->Args({8,128,512});
  b->Args({8,128,1024});
  b->Args({8,512,8});
  b->Args({8,512,32});
  b->Args({8,512,128});
  b->Args({8,512,512});
  b->Args({8,512,1024});
  b->Args({8,1024,8});
  b->Args({8,1024,32});
  b->Args({8,1024,128});
  b->Args({8,1024,512});
  b->Args({8,1024,1024});
  b->Args({32,8,8});
  b->Args({32,8,32});
  b->Args({32,8,128});
  b->Args({32,8,512});
  b->Args({32,8,1024});
  b->Args({32,32,8});
  b->Args({32,32,32});
  b->Args({32,32,128});
  b->Args({32,32,512});
  b->Args({32,32,1024});
  b->Args({32,128,8});
  b->Args({32,128,32});
  b->Args({32,128,128});
  b->Args({32,128,512});
  b->Args({32,128,1024});
  b->Args({32,512,8});
  b->Args({32,512,32});
  b->Args({32,512,128});
  b->Args({32,512,512});
  b->Args({32,512,1024});
  b->Args({32,1024,8});
  b->Args({32,1024,32});
  b->Args({32,1024,128});
  b->Args({32,1024,512});
  b->Args({32,1024,1024});
  b->Args({128,8,8});
  b->Args({128,8,32});
  b->Args({128,8,128});
  b->Args({128,8,512});
  b->Args({128,8,1024});
  b->Args({128,32,8});
  b->Args({128,32,32});
  b->Args({128,32,128});
  b->Args({128,32,512});
  b->Args({128,32,1024});
  b->Args({128,128,8});
  b->Args({128,128,32});
  b->Args({128,128,128});
  b->Args({128,128,512});
  b->Args({128,128,1024});
  b->Args({128,512,8});
  b->Args({128,512,32});
  b->Args({128,512,128});
  b->Args({128,512,512});
  b->Args({128,512,1024});
  b->Args({128,1024,8});
  b->Args({128,1024,32});
  b->Args({128,1024,128});
  b->Args({128,1024,512});
  b->Args({128,1024,1024});
  b->Args({512,8,8});
  b->Args({512,8,32});
  b->Args({512,8,128});
  b->Args({512,8,512});
  b->Args({512,8,1024});
  b->Args({512,32,8});
  b->Args({512,32,32});
  b->Args({512,32,128});
  b->Args({512,32,512});
  b->Args({512,32,1024});
  b->Args({512,128,8});
  b->Args({512,128,32});
  b->Args({512,128,128});
  b->Args({512,128,512});
  b->Args({512,128,1024});
  b->Args({512,512,8});
  b->Args({512,512,32});
  b->Args({512,512,128});
  b->Args({512,512,512});
  b->Args({512,512,1024});
  b->Args({512,1024,8});
  b->Args({512,1024,32});
  b->Args({512,1024,128});
  b->Args({512,1024,512});
  b->Args({512,1024,1024});
  b->Args({1024,8,8});
  b->Args({1024,8,32});
  b->Args({1024,8,128});
  b->Args({1024,8,512});
  b->Args({1024,8,1024});
  b->Args({1024,32,8});
  b->Args({1024,32,32});
  b->Args({1024,32,128});
  b->Args({1024,32,512});
  b->Args({1024,32,1024});
  b->Args({1024,128,8});
  b->Args({1024,128,32});
  b->Args({1024,128,128});
  b->Args({1024,128,512});
  b->Args({1024,128,1024});
  b->Args({1024,512,8});
  b->Args({1024,512,32});
  b->Args({1024,512,128});
  b->Args({1024,512,512});
  b->Args({1024,512,1024});
  b->Args({1024,1024,8});
  b->Args({1024,1024,32});
  b->Args({1024,1024,128});
  b->Args({1024,1024,512});
  b->Args({1024,1024,1024});
}

//BENCHMARK(MatrixMul1)->Apply(CustomArgs);
// BENCHMARK(MatrixMul2)->Apply(CustomArgs);
// BENCHMARK(MatrixMul3)->Apply(CustomArgs);
// BENCHMARK(MatrixMul4)->Apply(CustomArgs);
// BENCHMARK(MatrixMul5)->Apply(CustomArgs);
// BENCHMARK(MatrixMul6)->Apply(CustomArgs);
// BENCHMARK(MatrixMul7)->Apply(CustomArgs);
BENCHMARK(MatrixMul8)->Apply(CustomArgs);


BENCHMARK_MAIN();
// MatrixMul2/8/8/8                 201 ns          201 ns      3486160 items_per_second=4.97363M/s
// MatrixMul2/8/8/32                269 ns          269 ns      2594662 items_per_second=3.71191M/s
// MatrixMul2/8/8/128               540 ns          540 ns      1293122 items_per_second=1.85051M/s
// MatrixMul2/8/8/512              1939 ns         1939 ns       361183 items_per_second=515.783k/s
// MatrixMul2/8/8/1024             4414 ns         4414 ns       158681 items_per_second=226.563k/s
// MatrixMul2/8/32/8                785 ns          785 ns       891322 items_per_second=1.27453M/s
// MatrixMul2/8/32/32              1063 ns         1063 ns       658184 items_per_second=940.682k/s
// MatrixMul2/8/32/128             2143 ns         2142 ns       326845 items_per_second=466.753k/s
// MatrixMul2/8/32/512             8505 ns         8504 ns        80453 items_per_second=117.586k/s
// MatrixMul2/8/32/1024           17899 ns        17895 ns        39040 items_per_second=55.8804k/s
// MatrixMul2/8/128/8              3074 ns         3074 ns       227642 items_per_second=325.345k/s
// MatrixMul2/8/128/32             3932 ns         3932 ns       178045 items_per_second=254.344k/s
// MatrixMul2/8/128/128            8732 ns         8731 ns        79897 items_per_second=114.54k/s
// MatrixMul2/8/128/512           26763 ns        26758 ns        26097 items_per_second=37.3723k/s
// MatrixMul2/8/128/1024          86090 ns        86081 ns         8107 items_per_second=11.617k/s
// MatrixMul2/8/512/8             12428 ns        12426 ns        56477 items_per_second=80.4738k/s
// MatrixMul2/8/512/32            15836 ns        15833 ns        44728 items_per_second=63.1586k/s
// MatrixMul2/8/512/128           35016 ns        35011 ns        19971 items_per_second=28.5621k/s
// MatrixMul2/8/512/512          108050 ns       108033 ns         6475 items_per_second=9.25642k/s
// MatrixMul2/8/512/1024         351944 ns       351906 ns         1989 items_per_second=2.84166k/s
// MatrixMul2/8/1024/8            26157 ns        26152 ns        26925 items_per_second=38.2379k/s
// MatrixMul2/8/1024/32           32488 ns        32485 ns        21546 items_per_second=30.7838k/s
// MatrixMul2/8/1024/128          76307 ns        76291 ns         9098 items_per_second=13.1077k/s
// MatrixMul2/8/1024/512         332160 ns       332127 ns         2107 items_per_second=3.0109k/s
// MatrixMul2/8/1024/1024       1197256 ns      1197256 ns          699 items_per_second=835.243/s
// MatrixMul2/32/8/8                819 ns          819 ns       855697 items_per_second=1.22143M/s
// MatrixMul2/32/8/32              1035 ns         1035 ns       670321 items_per_second=966.001k/s
// MatrixMul2/32/8/128             2215 ns         2215 ns       322854 items_per_second=451.423k/s
// MatrixMul2/32/8/512             9182 ns         9181 ns        76253 items_per_second=108.915k/s
// MatrixMul2/32/8/1024           18383 ns        18382 ns        38107 items_per_second=54.4025k/s
// MatrixMul2/32/32/8              3185 ns         3185 ns       220472 items_per_second=313.99k/s
// MatrixMul2/32/32/32             4073 ns         4073 ns       171939 items_per_second=245.544k/s
// MatrixMul2/32/32/128            8378 ns         8377 ns        83473 items_per_second=119.376k/s
// MatrixMul2/32/32/512           40266 ns        40260 ns        17670 items_per_second=24.8385k/s
// MatrixMul2/32/32/1024          81897 ns        81884 ns         8560 items_per_second=12.2124k/s
// MatrixMul2/32/128/8            12498 ns        12494 ns        55108 items_per_second=80.0362k/s
// MatrixMul2/32/128/32           15925 ns        15922 ns        44044 items_per_second=62.8051k/s
// MatrixMul2/32/128/128          34060 ns        34055 ns        20548 items_per_second=29.3643k/s
// MatrixMul2/32/128/512         161493 ns       161477 ns         4372 items_per_second=6.19284k/s
// MatrixMul2/32/128/1024        324731 ns       324672 ns         2112 items_per_second=3.08003k/s
// MatrixMul2/32/512/8            51212 ns        51206 ns        13569 items_per_second=19.5288k/s
// MatrixMul2/32/512/32           66364 ns        66357 ns        10534 items_per_second=15.07k/s
// MatrixMul2/32/512/128         143604 ns       143580 ns         4857 items_per_second=6.96475k/s
// MatrixMul2/32/512/512         601657 ns       601577 ns         1153 items_per_second=1.6623k/s
// MatrixMul2/32/512/1024       1181941 ns      1181698 ns          592 items_per_second=846.24/s
// MatrixMul2/32/1024/8          104750 ns       104736 ns         6661 items_per_second=9.54779k/s
// MatrixMul2/32/1024/32         134460 ns       134445 ns         5191 items_per_second=7.43801k/s
// MatrixMul2/32/1024/128        286943 ns       286926 ns         2443 items_per_second=3.48522k/s
// MatrixMul2/32/1024/512       1214115 ns      1214003 ns          579 items_per_second=823.721/s
// MatrixMul2/32/1024/1024      2515285 ns      2514909 ns          280 items_per_second=397.629/s
// MatrixMul2/128/8/8              2960 ns         2959 ns       236298 items_per_second=337.904k/s
// MatrixMul2/128/8/32             3835 ns         3835 ns       182645 items_per_second=260.78k/s
// MatrixMul2/128/8/128            9408 ns         9407 ns        74398 items_per_second=106.305k/s
// MatrixMul2/128/8/512           31173 ns        31170 ns        22298 items_per_second=32.0824k/s
// MatrixMul2/128/8/1024          85939 ns        85925 ns         8062 items_per_second=11.638k/s
// MatrixMul2/128/32/8            11832 ns        11830 ns        58720 items_per_second=84.5318k/s
// MatrixMul2/128/32/32           15417 ns        15415 ns        45093 items_per_second=64.8734k/s
// MatrixMul2/128/32/128          37624 ns        37616 ns        18708 items_per_second=26.5844k/s
// MatrixMul2/128/32/512         163045 ns       163012 ns         4272 items_per_second=6.1345k/s
// MatrixMul2/128/32/1024        351947 ns       351876 ns         2003 items_per_second=2.84191k/s
// MatrixMul2/128/128/8           51063 ns        51048 ns        14477 items_per_second=19.5895k/s
// MatrixMul2/128/128/32          65881 ns        65874 ns         9197 items_per_second=15.1805k/s
// MatrixMul2/128/128/128        166405 ns       166387 ns         4261 items_per_second=6.01008k/s
// MatrixMul2/128/128/512        618034 ns       617953 ns         1123 items_per_second=1.61825k/s
// MatrixMul2/128/128/1024      1318357 ns      1318330 ns          538 items_per_second=758.535/s
// MatrixMul2/128/512/8          227221 ns       227207 ns         3071 items_per_second=4.40126k/s
// MatrixMul2/128/512/32         293255 ns       293105 ns         2449 items_per_second=3.41175k/s
// MatrixMul2/128/512/128        673640 ns       673560 ns         1055 items_per_second=1.48465k/s
// MatrixMul2/128/512/512       2497658 ns      2497436 ns          281 items_per_second=400.411/s
// MatrixMul2/128/512/1024      5202609 ns      5202003 ns          134 items_per_second=192.234/s
// MatrixMul2/128/1024/8         521363 ns       521277 ns         1347 items_per_second=1.91836k/s
// MatrixMul2/128/1024/32        619695 ns       619645 ns         1121 items_per_second=1.61383k/s
// MatrixMul2/128/1024/128      1338649 ns      1338410 ns          524 items_per_second=747.155/s
// MatrixMul2/128/1024/512      5091419 ns      5090597 ns          107 items_per_second=196.441/s
// MatrixMul2/128/1024/1024    10328225 ns     10326049 ns           68 items_per_second=96.8425/s
// MatrixMul2/512/8/8             11737 ns        11735 ns        59648 items_per_second=85.217k/s
// MatrixMul2/512/8/32            16401 ns        16400 ns        42683 items_per_second=60.9745k/s
// MatrixMul2/512/8/128           38134 ns        38128 ns        18361 items_per_second=26.2276k/s
// MatrixMul2/512/8/512          148317 ns       148313 ns         4704 items_per_second=6.74251k/s
// MatrixMul2/512/8/1024         366452 ns       366399 ns         1909 items_per_second=2.72927k/s
// MatrixMul2/512/32/8            48034 ns        48024 ns        14592 items_per_second=20.8229k/s
// MatrixMul2/512/32/32           72246 ns        72237 ns         7806 items_per_second=13.8433k/s
// MatrixMul2/512/32/128         186458 ns       186444 ns         3753 items_per_second=5.36354k/s
// MatrixMul2/512/32/512         764964 ns       764913 ns         1010 items_per_second=1.30734k/s
// MatrixMul2/512/32/1024       1387977 ns      1387784 ns          407 items_per_second=720.573/s
// MatrixMul2/512/128/8          194498 ns       194476 ns         3577 items_per_second=5.14202k/s
// MatrixMul2/512/128/32         293130 ns       293075 ns         2389 items_per_second=3.4121k/s
// MatrixMul2/512/128/128        665072 ns       664974 ns         1043 items_per_second=1.50382k/s
// MatrixMul2/512/128/512       2815491 ns      2815179 ns          247 items_per_second=355.217/s
// MatrixMul2/512/128/1024      5601519 ns      5601480 ns          120 items_per_second=178.524/s
// MatrixMul2/512/512/8         1041580 ns      1041403 ns          666 items_per_second=960.244/s
// MatrixMul2/512/512/32        1278803 ns      1278667 ns          548 items_per_second=782.064/s
// MatrixMul2/512/512/128       2682984 ns      2682526 ns          261 items_per_second=372.783/s
// MatrixMul2/512/512/512      11030348 ns     11028498 ns           63 items_per_second=90.6742/s
// MatrixMul2/512/512/1024     23636872 ns     23632677 ns           30 items_per_second=42.3143/s
// MatrixMul2/512/1024/8        2177979 ns      2177661 ns          320 items_per_second=459.208/s
// MatrixMul2/512/1024/32       2663849 ns      2663610 ns          263 items_per_second=375.43/s
// MatrixMul2/512/1024/128      5592231 ns      5591096 ns          124 items_per_second=178.856/s
// MatrixMul2/512/1024/512     23049393 ns     23046298 ns           30 items_per_second=43.3909/s
// MatrixMul2/512/1024/1024    52924977 ns     52916075 ns           13 items_per_second=18.8978/s
// MatrixMul2/1024/8/8            23584 ns        23582 ns        29671 items_per_second=42.406k/s
// MatrixMul2/1024/8/32           32549 ns        32542 ns        21504 items_per_second=30.7292k/s
// MatrixMul2/1024/8/128          88361 ns        88349 ns         7899 items_per_second=11.3187k/s
// MatrixMul2/1024/8/512         359118 ns       359063 ns         1947 items_per_second=2.78503k/s
// MatrixMul2/1024/8/1024       1839544 ns      1838961 ns          374 items_per_second=543.785/s
// MatrixMul2/1024/32/8           95356 ns        95346 ns         7293 items_per_second=10.4881k/s
// MatrixMul2/1024/32/32         144272 ns       144256 ns         4846 items_per_second=6.93211k/s
// MatrixMul2/1024/32/128        385486 ns       385440 ns         1816 items_per_second=2.59444k/s
// MatrixMul2/1024/32/512       1407139 ns      1406856 ns          497 items_per_second=710.805/s
// MatrixMul2/1024/32/1024      8063347 ns      8062095 ns           79 items_per_second=124.037/s
// MatrixMul2/1024/128/8         403778 ns       403710 ns         1732 items_per_second=2.47703k/s
// MatrixMul2/1024/128/32        593749 ns       593668 ns         1176 items_per_second=1.68444k/s
// MatrixMul2/1024/128/128      1372291 ns      1372106 ns          510 items_per_second=728.807/s
// MatrixMul2/1024/128/512      5559007 ns      5558140 ns          125 items_per_second=179.916/s
// MatrixMul2/1024/128/1024    32366748 ns     32357863 ns           21 items_per_second=30.9044/s
// MatrixMul2/1024/512/8        2119713 ns      2119465 ns          327 items_per_second=471.817/s
// MatrixMul2/1024/512/32       2590881 ns      2590598 ns          270 items_per_second=386.011/s
// MatrixMul2/1024/512/128      5642217 ns      5640990 ns          124 items_per_second=177.274/s
// MatrixMul2/1024/512/512     28760475 ns     28756514 ns           24 items_per_second=34.7747/s
// MatrixMul2/1024/512/1024   135398704 ns    135358904 ns            5 items_per_second=7.38777/s
// MatrixMul2/1024/1024/8      15043902 ns     15041264 ns           47 items_per_second=66.4838/s
// MatrixMul2/1024/1024/32     17143722 ns     17140581 ns           41 items_per_second=58.3411/s
// MatrixMul2/1024/1024/128    27965489 ns     27961461 ns           25 items_per_second=35.7635/s
// MatrixMul2/1024/1024/512    74351366 ns     74338315 ns            9 items_per_second=13.452/s
// MatrixMul2/1024/1024/1024  296663505 ns    296599647 ns            3 items_per_second=3.37155/s
// MatrixMul3/8/8/8                 200 ns          200 ns      3497558 items_per_second=4.99611M/s
// MatrixMul3/8/8/32                756 ns          756 ns       924783 items_per_second=1.32296M/s
// MatrixMul3/8/8/128              3014 ns         3013 ns       232222 items_per_second=331.842k/s
// MatrixMul3/8/8/512             11901 ns        11900 ns        58797 items_per_second=84.037k/s
// MatrixMul3/8/8/1024            24489 ns        24486 ns        28578 items_per_second=40.8396k/s
// MatrixMul3/8/32/8                878 ns          878 ns       797320 items_per_second=1.1394M/s
// MatrixMul3/8/32/32              3454 ns         3453 ns       202676 items_per_second=289.58k/s
// MatrixMul3/8/32/128            13716 ns        13714 ns        50987 items_per_second=72.9173k/s
// MatrixMul3/8/32/512           145575 ns       145555 ns         4803 items_per_second=6.87025k/s
// MatrixMul3/8/32/1024          570386 ns       570272 ns         1235 items_per_second=1.75355k/s
// MatrixMul3/8/128/8              5273 ns         5272 ns       132465 items_per_second=189.665k/s
// MatrixMul3/8/128/32            21079 ns        21078 ns        33142 items_per_second=47.4438k/s
// MatrixMul3/8/128/128          122578 ns       122561 ns         5713 items_per_second=8.15919k/s
// MatrixMul3/8/128/512          979023 ns       978895 ns          713 items_per_second=1.02156k/s
// MatrixMul3/8/128/1024        5309828 ns      5309024 ns          132 items_per_second=188.359/s
// MatrixMul3/8/512/8             23311 ns        23309 ns        29928 items_per_second=42.9014k/s
// MatrixMul3/8/512/32            93284 ns        93271 ns         7506 items_per_second=10.7214k/s
// MatrixMul3/8/512/128          513259 ns       513211 ns         1347 items_per_second=1.94852k/s
// MatrixMul3/8/512/512         4230048 ns      4229570 ns          166 items_per_second=236.431/s
// MatrixMul3/8/512/1024       24240656 ns     24237343 ns           29 items_per_second=41.2586/s
// MatrixMul3/8/1024/8            47248 ns        47241 ns        14818 items_per_second=21.1679k/s
// MatrixMul3/8/1024/32          189458 ns       189447 ns         3696 items_per_second=5.27852k/s
// MatrixMul3/8/1024/128        1178580 ns      1178372 ns          594 items_per_second=848.629/s
// MatrixMul3/8/1024/512        8804930 ns      8803104 ns           79 items_per_second=113.596/s
// MatrixMul3/8/1024/1024      54997279 ns     54990201 ns           12 items_per_second=18.1851/s
// MatrixMul3/32/8/8                779 ns          779 ns       897283 items_per_second=1.28339M/s
// MatrixMul3/32/8/32              3003 ns         3002 ns       233059 items_per_second=333.06k/s
// MatrixMul3/32/8/128            12041 ns        12040 ns        58117 items_per_second=83.0568k/s
// MatrixMul3/32/8/512            47625 ns        47621 ns        14701 items_per_second=20.9991k/s
// MatrixMul3/32/8/1024           97901 ns        97888 ns         7140 items_per_second=10.2157k/s
// MatrixMul3/32/32/8              3496 ns         3496 ns       200206 items_per_second=286.03k/s
// MatrixMul3/32/32/32            13809 ns        13808 ns        50682 items_per_second=72.4241k/s
// MatrixMul3/32/32/128           54887 ns        54878 ns        12700 items_per_second=18.2221k/s
// MatrixMul3/32/32/512          596812 ns       596736 ns         1178 items_per_second=1.67578k/s
// MatrixMul3/32/32/1024        2745985 ns      2745346 ns          254 items_per_second=364.253/s
// MatrixMul3/32/128/8            21219 ns        21215 ns        33165 items_per_second=47.1369k/s
// MatrixMul3/32/128/32           84373 ns        84358 ns         8292 items_per_second=11.8542k/s
// MatrixMul3/32/128/128         492103 ns       492053 ns         1392 items_per_second=2.0323k/s
// MatrixMul3/32/128/512        4003531 ns      4002863 ns          175 items_per_second=249.821/s
// MatrixMul3/32/128/1024      21247444 ns     21244881 ns           33 items_per_second=47.0702/s
// MatrixMul3/32/512/8            93033 ns        93024 ns         7516 items_per_second=10.7499k/s
// MatrixMul3/32/512/32          378065 ns       378035 ns         1852 items_per_second=2.64526k/s
// MatrixMul3/32/512/128        2029112 ns      2028926 ns          329 items_per_second=492.872/s
// MatrixMul3/32/512/512       17213907 ns     17211540 ns           40 items_per_second=58.1006/s
// MatrixMul3/32/512/1024      97961061 ns     97947114 ns            7 items_per_second=10.2096/s
// MatrixMul3/32/1024/8          189107 ns       189073 ns         3703 items_per_second=5.28897k/s
// MatrixMul3/32/1024/32         763196 ns       763099 ns          917 items_per_second=1.31045k/s
// MatrixMul3/32/1024/128       4679283 ns      4678499 ns          149 items_per_second=213.744/s
// MatrixMul3/32/1024/512      35239582 ns     35233733 ns           20 items_per_second=28.3819/s
// MatrixMul3/32/1024/1024    219337086 ns    219278120 ns            3 items_per_second=4.56042/s
// MatrixMul3/128/8/8              3095 ns         3094 ns       226201 items_per_second=323.16k/s
// MatrixMul3/128/8/32            11989 ns        11986 ns        58365 items_per_second=83.4281k/s
// MatrixMul3/128/8/128           48175 ns        48170 ns        14532 items_per_second=20.7599k/s
// MatrixMul3/128/8/512          190497 ns       190467 ns         3673 items_per_second=5.25025k/s
// MatrixMul3/128/8/1024         469428 ns       469366 ns         1491 items_per_second=2.13053k/s
// MatrixMul3/128/32/8            14024 ns        14022 ns        49849 items_per_second=71.3173k/s
// MatrixMul3/128/32/32           55170 ns        55164 ns        12655 items_per_second=18.1279k/s
// MatrixMul3/128/32/128         219627 ns       219589 ns         3187 items_per_second=4.55397k/s
// MatrixMul3/128/32/512        2415818 ns      2415506 ns          290 items_per_second=413.992/s
// MatrixMul3/128/32/1024      10912051 ns     10910181 ns           64 items_per_second=91.6575/s
// MatrixMul3/128/128/8           84462 ns        84457 ns         8281 items_per_second=11.8404k/s
// MatrixMul3/128/128/32         337430 ns       337398 ns         2073 items_per_second=2.96386k/s
// MatrixMul3/128/128/128       2081296 ns      2081155 ns          335 items_per_second=480.502/s
// MatrixMul3/128/128/512      15870115 ns     15868734 ns           44 items_per_second=63.017/s
// MatrixMul3/128/128/1024     84751920 ns     84735707 ns            8 items_per_second=11.8014/s
// MatrixMul3/128/512/8          372352 ns       372293 ns         1881 items_per_second=2.68605k/s
// MatrixMul3/128/512/32        1512872 ns      1512578 ns          463 items_per_second=661.123/s
// MatrixMul3/128/512/128       8422033 ns      8421081 ns           81 items_per_second=118.75/s
// MatrixMul3/128/512/512      67694926 ns     67685694 ns           10 items_per_second=14.7742/s
// MatrixMul3/128/512/1024    389355162 ns    389322963 ns            2 items_per_second=2.56856/s
// MatrixMul3/128/1024/8         758105 ns       758020 ns          923 items_per_second=1.31923k/s
// MatrixMul3/128/1024/32       3054155 ns      3053954 ns          229 items_per_second=327.444/s
// MatrixMul3/128/1024/128     18645373 ns     18643169 ns           37 items_per_second=53.6389/s
// MatrixMul3/128/1024/512    147302049 ns    147283671 ns            5 items_per_second=6.78962/s
// MatrixMul3/128/1024/1024   885382971 ns    885231999 ns            1 items_per_second=1.12965/s
// MatrixMul3/512/8/8             12387 ns        12385 ns        56587 items_per_second=80.7416k/s
// MatrixMul3/512/8/32            48218 ns        48211 ns        14539 items_per_second=20.7423k/s
// MatrixMul3/512/8/128          193520 ns       193489 ns         3619 items_per_second=5.16825k/s
// MatrixMul3/512/8/512          767291 ns       767060 ns          914 items_per_second=1.30368k/s
// MatrixMul3/512/8/1024        1884804 ns      1884348 ns          372 items_per_second=530.688/s
// MatrixMul3/512/32/8            56262 ns        56257 ns        12391 items_per_second=17.7757k/s
// MatrixMul3/512/32/32          225410 ns       225372 ns         3107 items_per_second=4.43711k/s
// MatrixMul3/512/32/128         923964 ns       923786 ns          758 items_per_second=1.0825k/s
// MatrixMul3/512/32/512        9530955 ns      9529541 ns           73 items_per_second=104.937/s
// MatrixMul3/512/32/1024      40477854 ns     40466026 ns           17 items_per_second=24.7121/s
// MatrixMul3/512/128/8          339353 ns       339282 ns         2063 items_per_second=2.9474k/s
// MatrixMul3/512/128/32        1358515 ns      1358281 ns          515 items_per_second=736.224/s
// MatrixMul3/512/128/128       7852726 ns      7851421 ns           83 items_per_second=127.365/s
// MatrixMul3/512/128/512      62371917 ns     62359317 ns           11 items_per_second=16.0361/s
// MatrixMul3/512/128/1024    313307295 ns    313264644 ns            2 items_per_second=3.19219/s
// MatrixMul3/512/512/8         1497755 ns      1497429 ns          467 items_per_second=667.811/s
// MatrixMul3/512/512/32        5989694 ns      5988748 ns          117 items_per_second=166.98/s
// MatrixMul3/512/512/128      31951094 ns     31945874 ns           22 items_per_second=31.3029/s
// MatrixMul3/512/512/512     280770291 ns    280720272 ns            2 items_per_second=3.56226/s
// MatrixMul3/512/512/1024   1553837935 ns   1553513728 ns            1 items_per_second=0.643702/s
// MatrixMul3/512/1024/8        3046006 ns      3045648 ns          230 items_per_second=328.337/s
// MatrixMul3/512/1024/32      12167113 ns     12165310 ns           57 items_per_second=82.2009/s
// MatrixMul3/512/1024/128     75591927 ns     75584599 ns            9 items_per_second=13.2302/s
// MatrixMul3/512/1024/512    684187991 ns    684086414 ns            1 items_per_second=1.4618/s
// MatrixMul3/512/1024/1024  3531298724 ns   3530611027 ns            1 items_per_second=0.283237/s
// MatrixMul3/1024/8/8            24822 ns        24819 ns        28187 items_per_second=40.2914k/s
// MatrixMul3/1024/8/32           96478 ns        96460 ns         7261 items_per_second=10.367k/s
// MatrixMul3/1024/8/128         387299 ns       387231 ns         1804 items_per_second=2.58243k/s
// MatrixMul3/1024/8/512        1535134 ns      1534985 ns          456 items_per_second=651.472/s
// MatrixMul3/1024/8/1024       3779275 ns      3778059 ns          185 items_per_second=264.686/s
// MatrixMul3/1024/32/8          112621 ns       112610 ns         6213 items_per_second=8.88024k/s
// MatrixMul3/1024/32/32         450453 ns       450393 ns         1554 items_per_second=2.22028k/s
// MatrixMul3/1024/32/128       1863686 ns      1863544 ns          379 items_per_second=536.612/s
// MatrixMul3/1024/32/512      19060815 ns     19058373 ns           36 items_per_second=52.4704/s
// MatrixMul3/1024/32/1024     80952987 ns     80940423 ns            9 items_per_second=12.3548/s
// MatrixMul3/1024/128/8         679058 ns       678967 ns         1031 items_per_second=1.47283k/s
// MatrixMul3/1024/128/32       2718210 ns      2717767 ns          258 items_per_second=367.949/s
// MatrixMul3/1024/128/128     15675066 ns     15673358 ns           45 items_per_second=63.8025/s
// MatrixMul3/1024/128/512    119285010 ns    119266988 ns            6 items_per_second=8.38455/s
// MatrixMul3/1024/128/1024   674754361 ns    674699134 ns            1 items_per_second=1.48214/s
// MatrixMul3/1024/512/8        2996584 ns      2996258 ns          234 items_per_second=333.75/s
// MatrixMul3/1024/512/32      11980124 ns     11979117 ns           58 items_per_second=83.4786/s
// MatrixMul3/1024/512/128     63346462 ns     63338805 ns           11 items_per_second=15.7881/s
// MatrixMul3/1024/512/512    567836868 ns    567718676 ns            1 items_per_second=1.76144/s
// MatrixMul3/1024/512/1024  3129827480 ns   3129197281 ns            1 items_per_second=0.319571/s
// MatrixMul3/1024/1024/8       6125143 ns      6123387 ns          114 items_per_second=163.308/s
// MatrixMul3/1024/1024/32     24382995 ns     24379798 ns           29 items_per_second=41.0176/s
// MatrixMul3/1024/1024/128   151671808 ns    151648986 ns            5 items_per_second=6.59418/s
// MatrixMul3/1024/1024/512  1142750613 ns   1142490869 ns            1 items_per_second=0.875281/s
// MatrixMul3/1024/1024/1024 6925885524 ns   6923908049 ns            1 items_per_second=0.144427/s
// MatrixMul4/8/8/8                 168 ns          168 ns      4165456 items_per_second=5.94776M/s
// MatrixMul4/8/8/32                612 ns          612 ns      1142602 items_per_second=1.63439M/s
// MatrixMul4/8/8/128              2404 ns         2404 ns       291139 items_per_second=415.969k/s
// MatrixMul4/8/8/512              9464 ns         9463 ns        73983 items_per_second=105.68k/s
// MatrixMul4/8/8/1024            18884 ns        18880 ns        37067 items_per_second=52.9652k/s
// MatrixMul4/8/32/8                541 ns          540 ns      1293971 items_per_second=1.85023M/s
// MatrixMul4/8/32/32              2137 ns         2136 ns       327183 items_per_second=468.127k/s
// MatrixMul4/8/32/128             8383 ns         8381 ns        83346 items_per_second=119.313k/s
// MatrixMul4/8/32/512            33427 ns        33421 ns        20938 items_per_second=29.9215k/s
// MatrixMul4/8/32/1024           66887 ns        66878 ns        10476 items_per_second=14.9526k/s
// MatrixMul4/8/128/8              2058 ns         2058 ns       339396 items_per_second=485.941k/s
// MatrixMul4/8/128/32             8074 ns         8072 ns        86526 items_per_second=123.886k/s
// MatrixMul4/8/128/128           33025 ns        33020 ns        21195 items_per_second=30.2851k/s
// MatrixMul4/8/128/512          131578 ns       131537 ns         5320 items_per_second=7.60245k/s
// MatrixMul4/8/128/1024         263392 ns       263341 ns         2657 items_per_second=3.79736k/s
// MatrixMul4/8/512/8              8246 ns         8244 ns        84466 items_per_second=121.295k/s
// MatrixMul4/8/512/32            33459 ns        33454 ns        20899 items_per_second=29.8919k/s
// MatrixMul4/8/512/128          133908 ns       133883 ns         5206 items_per_second=7.46923k/s
// MatrixMul4/8/512/512          541139 ns       541030 ns         1278 items_per_second=1.84833k/s
// MatrixMul4/8/512/1024        1083005 ns      1082726 ns          646 items_per_second=923.595/s
// MatrixMul4/8/1024/8            16733 ns        16727 ns        41809 items_per_second=59.7846k/s
// MatrixMul4/8/1024/32           66078 ns        66070 ns        10566 items_per_second=15.1355k/s
// MatrixMul4/8/1024/128         265127 ns       265071 ns         2639 items_per_second=3.77257k/s
// MatrixMul4/8/1024/512        1071754 ns      1071604 ns          654 items_per_second=933.181/s
// MatrixMul4/8/1024/1024       2344056 ns      2343039 ns          300 items_per_second=426.796/s
// MatrixMul4/32/8/8                651 ns          651 ns      1070857 items_per_second=1.53705M/s
// MatrixMul4/32/8/32              2415 ns         2415 ns       289418 items_per_second=414.134k/s
// MatrixMul4/32/8/128             9598 ns         9596 ns        72924 items_per_second=104.213k/s
// MatrixMul4/32/8/512            37826 ns        37820 ns        18508 items_per_second=26.441k/s
// MatrixMul4/32/8/1024           75517 ns        75503 ns         9263 items_per_second=13.2444k/s
// MatrixMul4/32/32/8              2144 ns         2143 ns       326535 items_per_second=466.568k/s
// MatrixMul4/32/32/32             8528 ns         8527 ns        81955 items_per_second=117.278k/s
// MatrixMul4/32/32/128           33530 ns        33525 ns        20873 items_per_second=29.8283k/s
// MatrixMul4/32/32/512          133733 ns       133710 ns         5231 items_per_second=7.47887k/s
// MatrixMul4/32/32/1024         267336 ns       267260 ns         2620 items_per_second=3.74167k/s
// MatrixMul4/32/128/8             8248 ns         8247 ns        85027 items_per_second=121.263k/s
// MatrixMul4/32/128/32           32352 ns        32344 ns        20877 items_per_second=30.918k/s
// MatrixMul4/32/128/128         132291 ns       132259 ns         5288 items_per_second=7.56095k/s
// MatrixMul4/32/128/512         527232 ns       527171 ns         1329 items_per_second=1.89692k/s
// MatrixMul4/32/128/1024       1054271 ns      1054168 ns          662 items_per_second=948.615/s
// MatrixMul4/32/512/8            33032 ns        33027 ns        21176 items_per_second=30.2785k/s
// MatrixMul4/32/512/32          133902 ns       133885 ns         5226 items_per_second=7.4691k/s
// MatrixMul4/32/512/128         535056 ns       534971 ns         1275 items_per_second=1.86926k/s
// MatrixMul4/32/512/512        2165303 ns      2165044 ns          323 items_per_second=461.884/s
// MatrixMul4/32/512/1024       4322014 ns      4321168 ns          162 items_per_second=231.419/s
// MatrixMul4/32/1024/8           66564 ns        66555 ns        10472 items_per_second=15.0251k/s
// MatrixMul4/32/1024/32         266662 ns       266617 ns         2651 items_per_second=3.75069k/s
// MatrixMul4/32/1024/128       1062369 ns      1062233 ns          657 items_per_second=941.413/s
// MatrixMul4/32/1024/512       4275380 ns      4274782 ns          164 items_per_second=233.93/s
// MatrixMul4/32/1024/1024      9314809 ns      9313484 ns           75 items_per_second=107.371/s
// MatrixMul4/128/8/8              2596 ns         2596 ns       270542 items_per_second=385.208k/s
// MatrixMul4/128/8/32             9896 ns         9895 ns        72152 items_per_second=101.063k/s
// MatrixMul4/128/8/128           39866 ns        39861 ns        18270 items_per_second=25.0874k/s
// MatrixMul4/128/8/512          151324 ns       151302 ns         4604 items_per_second=6.60932k/s
// MatrixMul4/128/8/1024         302893 ns       302855 ns         2310 items_per_second=3.30191k/s
// MatrixMul4/128/32/8             8561 ns         8559 ns        81457 items_per_second=116.83k/s
// MatrixMul4/128/32/32           34104 ns        34098 ns        20198 items_per_second=29.327k/s
// MatrixMul4/128/32/128         134232 ns       134207 ns         5215 items_per_second=7.45119k/s
// MatrixMul4/128/32/512         536141 ns       536056 ns         1282 items_per_second=1.86548k/s
// MatrixMul4/128/32/1024       1069843 ns      1069690 ns          638 items_per_second=934.851/s
// MatrixMul4/128/128/8           32917 ns        32911 ns        21252 items_per_second=30.3849k/s
// MatrixMul4/128/128/32         129325 ns       129304 ns         5410 items_per_second=7.7337k/s
// MatrixMul4/128/128/128        529053 ns       528988 ns         1298 items_per_second=1.8904k/s
// MatrixMul4/128/128/512       2108895 ns      2108608 ns          332 items_per_second=474.246/s
// MatrixMul4/128/128/1024      4222667 ns      4222204 ns          166 items_per_second=236.843/s
// MatrixMul4/128/512/8          132235 ns       132219 ns         5293 items_per_second=7.56321k/s
// MatrixMul4/128/512/32         536090 ns       535973 ns         1302 items_per_second=1.86577k/s
// MatrixMul4/128/512/128       2142439 ns      2142178 ns          327 items_per_second=466.815/s
// MatrixMul4/128/512/512       8664459 ns      8663244 ns           81 items_per_second=115.43/s
// MatrixMul4/128/512/1024     17292623 ns     17290825 ns           40 items_per_second=57.8341/s
// MatrixMul4/128/1024/8         267065 ns       267022 ns         2621 items_per_second=3.74501k/s
// MatrixMul4/128/1024/32       1057762 ns      1057661 ns          660 items_per_second=945.483/s
// MatrixMul4/128/1024/128      4257784 ns      4256968 ns          164 items_per_second=234.909/s
// MatrixMul4/128/1024/512     17123075 ns     17120904 ns           41 items_per_second=58.4081/s
// MatrixMul4/128/1024/1024    37417219 ns     37391805 ns           19 items_per_second=26.7438/s
// MatrixMul4/512/8/8             10318 ns        10317 ns        67596 items_per_second=96.932k/s
// MatrixMul4/512/8/32            38541 ns        38534 ns        18166 items_per_second=25.9514k/s
// MatrixMul4/512/8/128          153301 ns       153284 ns         4566 items_per_second=6.52385k/s
// MatrixMul4/512/8/512          607039 ns       606966 ns         1148 items_per_second=1.64754k/s
// MatrixMul4/512/8/1024        1212200 ns      1211992 ns          577 items_per_second=825.088/s
// MatrixMul4/512/32/8            34230 ns        34225 ns        20458 items_per_second=29.2181k/s
// MatrixMul4/512/32/32          136449 ns       136429 ns         5129 items_per_second=7.32984k/s
// MatrixMul4/512/32/128         539326 ns       539246 ns         1301 items_per_second=1.85444k/s
// MatrixMul4/512/32/512        2141187 ns      2140822 ns          327 items_per_second=467.11/s
// MatrixMul4/512/32/1024       4276259 ns      4275881 ns          164 items_per_second=233.87/s
// MatrixMul4/512/128/8          131657 ns       131637 ns         5315 items_per_second=7.59662k/s
// MatrixMul4/512/128/32         517556 ns       517485 ns         1340 items_per_second=1.93242k/s
// MatrixMul4/512/128/128       2117988 ns      2117671 ns          331 items_per_second=472.217/s
// MatrixMul4/512/128/512       8435386 ns      8434220 ns           83 items_per_second=118.565/s
// MatrixMul4/512/128/1024     16933779 ns     16932033 ns           41 items_per_second=59.0597/s
// MatrixMul4/512/512/8          531813 ns       531710 ns         1300 items_per_second=1.88072k/s
// MatrixMul4/512/512/32        2149351 ns      2148932 ns          326 items_per_second=465.347/s
// MatrixMul4/512/512/128       8578675 ns      8576844 ns           81 items_per_second=116.593/s
// MatrixMul4/512/512/512      34660817 ns     34654921 ns           20 items_per_second=28.8559/s
// MatrixMul4/512/512/1024     69194886 ns     69173254 ns           10 items_per_second=14.4565/s
// MatrixMul4/512/1024/8        1073735 ns      1073558 ns          652 items_per_second=931.482/s
// MatrixMul4/512/1024/32       4243212 ns      4242175 ns          165 items_per_second=235.728/s
// MatrixMul4/512/1024/128     17079793 ns     17076658 ns           41 items_per_second=58.5595/s
// MatrixMul4/512/1024/512     68400907 ns     68387240 ns           10 items_per_second=14.6226/s
// MatrixMul4/512/1024/1024   150289974 ns    150232826 ns            5 items_per_second=6.65633/s
// MatrixMul4/1024/8/8            20638 ns        20635 ns        33893 items_per_second=48.4615k/s
// MatrixMul4/1024/8/32           77155 ns        77139 ns         9028 items_per_second=12.9636k/s
// MatrixMul4/1024/8/128         307412 ns       307362 ns         2277 items_per_second=3.2535k/s
// MatrixMul4/1024/8/512        1216616 ns      1215839 ns          574 items_per_second=822.477/s
// MatrixMul4/1024/8/1024       2426516 ns      2425701 ns          288 items_per_second=412.252/s
// MatrixMul4/1024/32/8           68476 ns        68465 ns        10213 items_per_second=14.6059k/s
// MatrixMul4/1024/32/32         273098 ns       273022 ns         2564 items_per_second=3.6627k/s
// MatrixMul4/1024/32/128       1075652 ns      1075365 ns          651 items_per_second=929.917/s
// MatrixMul4/1024/32/512       4573912 ns      4572586 ns          163 items_per_second=218.695/s
// MatrixMul4/1024/32/1024      8808338 ns      8807833 ns           76 items_per_second=113.535/s
// MatrixMul4/1024/128/8         264505 ns       264461 ns         2611 items_per_second=3.78127k/s
// MatrixMul4/1024/128/32       1040161 ns      1040007 ns          674 items_per_second=961.532/s
// MatrixMul4/1024/128/128      4250678 ns      4249787 ns          165 items_per_second=235.306/s
// MatrixMul4/1024/128/512     16901994 ns     16900667 ns           41 items_per_second=59.1693/s
// MatrixMul4/1024/128/1024    33862894 ns     33855964 ns           21 items_per_second=29.5369/s
// MatrixMul4/1024/512/8        1075209 ns      1075076 ns          655 items_per_second=930.167/s
// MatrixMul4/1024/512/32       4305407 ns      4305021 ns          162 items_per_second=232.287/s
// MatrixMul4/1024/512/128     17220173 ns     17217959 ns           41 items_per_second=58.0789/s
// MatrixMul4/1024/512/512     69420282 ns     69405437 ns           10 items_per_second=14.4081/s
// MatrixMul4/1024/512/1024   138369050 ns    138346553 ns            5 items_per_second=7.22822/s
// MatrixMul4/1024/1024/8       2223893 ns      2223149 ns          314 items_per_second=449.812/s
// MatrixMul4/1024/1024/32      8609483 ns      8608333 ns           81 items_per_second=116.167/s
// MatrixMul4/1024/1024/128    34264545 ns     34258251 ns           20 items_per_second=29.19/s
// MatrixMul4/1024/1024/512   137156183 ns    137130505 ns            5 items_per_second=7.29232/s
// MatrixMul4/1024/1024/1024  298596742 ns    298491649 ns            2 items_per_second=3.35018/s
// MatrixMul5/8/8/8                 165 ns          165 ns      4251600 items_per_second=6.07205M/s
// MatrixMul5/8/8/32                219 ns          219 ns      3200874 items_per_second=4.57153M/s
// MatrixMul5/8/8/128               419 ns          419 ns      1665182 items_per_second=2.38771M/s
// MatrixMul5/8/8/512              1363 ns         1362 ns       510211 items_per_second=734.043k/s
// MatrixMul5/8/8/1024             3561 ns         3561 ns       197501 items_per_second=280.828k/s
// MatrixMul5/8/32/8                631 ns          631 ns      1083773 items_per_second=1.58443M/s
// MatrixMul5/8/32/32               861 ns          861 ns       807593 items_per_second=1.16195M/s
// MatrixMul5/8/32/128             1649 ns         1648 ns       423360 items_per_second=606.617k/s
// MatrixMul5/8/32/512             8493 ns         8492 ns        82121 items_per_second=117.762k/s
// MatrixMul5/8/32/1024           17327 ns        17324 ns        40183 items_per_second=57.7231k/s
// MatrixMul5/8/128/8              2507 ns         2507 ns       278578 items_per_second=398.86k/s
// MatrixMul5/8/128/32             3248 ns         3247 ns       215588 items_per_second=307.951k/s
// MatrixMul5/8/128/128            6859 ns         6858 ns       101717 items_per_second=145.809k/s
// MatrixMul5/8/128/512           18901 ns        18897 ns        37056 items_per_second=52.9183k/s
// MatrixMul5/8/128/1024          82544 ns        82534 ns         8427 items_per_second=12.1162k/s
// MatrixMul5/8/512/8             10022 ns        10020 ns        69360 items_per_second=99.7967k/s
// MatrixMul5/8/512/32            12900 ns        12899 ns        53877 items_per_second=77.5252k/s
// MatrixMul5/8/512/128           27353 ns        27348 ns        25576 items_per_second=36.5653k/s
// MatrixMul5/8/512/512           77542 ns        77532 ns         8971 items_per_second=12.8979k/s
// MatrixMul5/8/512/1024         346346 ns       346285 ns         2021 items_per_second=2.88779k/s
// MatrixMul5/8/1024/8            19945 ns        19942 ns        35118 items_per_second=50.1464k/s
// MatrixMul5/8/1024/32           25736 ns        25733 ns        27207 items_per_second=38.8605k/s
// MatrixMul5/8/1024/128          71936 ns        71929 ns         9674 items_per_second=13.9026k/s
// MatrixMul5/8/1024/512         336589 ns       336551 ns         2079 items_per_second=2.97132k/s
// MatrixMul5/8/1024/1024        795179 ns       795047 ns          802 items_per_second=1.25779k/s
// MatrixMul5/32/8/8                659 ns          659 ns      1059258 items_per_second=1.51704M/s
// MatrixMul5/32/8/32               823 ns          823 ns       849168 items_per_second=1.21575M/s
// MatrixMul5/32/8/128             1623 ns         1623 ns       431477 items_per_second=616.207k/s
// MatrixMul5/32/8/512             7730 ns         7729 ns        90857 items_per_second=129.389k/s
// MatrixMul5/32/8/1024           15943 ns        15941 ns        43507 items_per_second=62.7301k/s
// MatrixMul5/32/32/8              2553 ns         2552 ns       274048 items_per_second=391.811k/s
// MatrixMul5/32/32/32             3263 ns         3262 ns       214666 items_per_second=306.532k/s
// MatrixMul5/32/32/128            6321 ns         6321 ns       110536 items_per_second=158.214k/s
// MatrixMul5/32/32/512           36708 ns        36702 ns        19069 items_per_second=27.2466k/s
// MatrixMul5/32/32/1024          75495 ns        75483 ns         9176 items_per_second=13.248k/s
// MatrixMul5/32/128/8            10044 ns        10041 ns        68678 items_per_second=99.5879k/s
// MatrixMul5/32/128/32           12944 ns        12942 ns        53607 items_per_second=77.2651k/s
// MatrixMul5/32/128/128          25907 ns        25904 ns        27026 items_per_second=38.6035k/s
// MatrixMul5/32/128/512         147561 ns       147550 ns         4744 items_per_second=6.77735k/s
// MatrixMul5/32/128/1024        303190 ns       303152 ns         2309 items_per_second=3.29868k/s
// MatrixMul5/32/512/8            40314 ns        40310 ns        17369 items_per_second=24.8077k/s
// MatrixMul5/32/512/32           52236 ns        52229 ns        13343 items_per_second=19.1465k/s
// MatrixMul5/32/512/128         136906 ns       136889 ns         4973 items_per_second=7.30517k/s
// MatrixMul5/32/512/512         531536 ns       531463 ns         1295 items_per_second=1.8816k/s
// MatrixMul5/32/512/1024       1113315 ns      1113089 ns          627 items_per_second=898.401/s
// MatrixMul5/32/1024/8           80280 ns        80267 ns         8632 items_per_second=12.4584k/s
// MatrixMul5/32/1024/32         103748 ns       103736 ns         6700 items_per_second=9.63984k/s
// MatrixMul5/32/1024/128        274024 ns       273988 ns         2561 items_per_second=3.64979k/s
// MatrixMul5/32/1024/512       1061866 ns      1061721 ns          658 items_per_second=941.867/s
// MatrixMul5/32/1024/1024      2346430 ns      2345899 ns          300 items_per_second=426.276/s
// MatrixMul5/128/8/8              2339 ns         2339 ns       299652 items_per_second=427.607k/s
// MatrixMul5/128/8/32             3070 ns         3069 ns       228108 items_per_second=325.824k/s
// MatrixMul5/128/8/128            6910 ns         6909 ns       100723 items_per_second=144.744k/s
// MatrixMul5/128/8/512           26896 ns        26892 ns        25899 items_per_second=37.1864k/s
// MatrixMul5/128/8/1024          77096 ns        77081 ns         9035 items_per_second=12.9733k/s
// MatrixMul5/128/32/8             9304 ns         9302 ns        75008 items_per_second=107.498k/s
// MatrixMul5/128/32/32           12269 ns        12268 ns        57038 items_per_second=81.5144k/s
// MatrixMul5/128/32/128          27604 ns        27599 ns        25032 items_per_second=36.2336k/s
// MatrixMul5/128/32/512         143704 ns       143685 ns         4857 items_per_second=6.95966k/s
// MatrixMul5/128/32/1024        310426 ns       310389 ns         2255 items_per_second=3.22176k/s
// MatrixMul5/128/128/8           37335 ns        37332 ns        18705 items_per_second=26.787k/s
// MatrixMul5/128/128/32          49583 ns        49577 ns        13560 items_per_second=20.1704k/s
// MatrixMul5/128/128/128        133875 ns       133853 ns         5230 items_per_second=7.47088k/s
// MatrixMul5/128/128/512        559952 ns       559890 ns         1245 items_per_second=1.78607k/s
// MatrixMul5/128/128/1024      1235603 ns      1235426 ns          566 items_per_second=809.437/s
// MatrixMul5/128/512/8          149186 ns       149174 ns         4681 items_per_second=6.7036k/s
// MatrixMul5/128/512/32         201575 ns       201545 ns         3170 items_per_second=4.96168k/s
// MatrixMul5/128/512/128        543941 ns       543876 ns         1281 items_per_second=1.83865k/s
// MatrixMul5/128/512/512       2168911 ns      2168560 ns          323 items_per_second=461.136/s
// MatrixMul5/128/512/1024      4887088 ns      4885990 ns          136 items_per_second=204.667/s
// MatrixMul5/128/1024/8         305505 ns       305463 ns         2315 items_per_second=3.27372k/s
// MatrixMul5/128/1024/32        404855 ns       404778 ns         1732 items_per_second=2.47049k/s
// MatrixMul5/128/1024/128      1091502 ns      1091357 ns          642 items_per_second=916.29/s
// MatrixMul5/128/1024/512      4343784 ns      4343245 ns          161 items_per_second=230.243/s
// MatrixMul5/128/1024/1024    10076247 ns     10074543 ns           70 items_per_second=99.2601/s
// MatrixMul5/512/8/8              9165 ns         9164 ns        76024 items_per_second=109.127k/s
// MatrixMul5/512/8/32            13482 ns        13479 ns        51908 items_per_second=74.1878k/s
// MatrixMul5/512/8/128           28392 ns        28387 ns        24656 items_per_second=35.2272k/s
// MatrixMul5/512/8/512          127178 ns       127156 ns         5509 items_per_second=7.86436k/s
// MatrixMul5/512/8/1024         315333 ns       315295 ns         2222 items_per_second=3.17164k/s
// MatrixMul5/512/32/8            36810 ns        36806 ns        19026 items_per_second=27.1692k/s
// MatrixMul5/512/32/32           56638 ns        56632 ns        12321 items_per_second=17.6577k/s
// MatrixMul5/512/32/128         141823 ns       141803 ns         4924 items_per_second=7.05203k/s
// MatrixMul5/512/32/512         613199 ns       613143 ns         1140 items_per_second=1.63094k/s
// MatrixMul5/512/32/1024       1280466 ns      1280252 ns          546 items_per_second=781.096/s
// MatrixMul5/512/128/8          145772 ns       145754 ns         4794 items_per_second=6.86088k/s
// MatrixMul5/512/128/32         225465 ns       225413 ns         3105 items_per_second=4.43631k/s
// MatrixMul5/512/128/128        561173 ns       561095 ns         1232 items_per_second=1.78223k/s
// MatrixMul5/512/128/512       2542747 ns      2542469 ns          275 items_per_second=393.318/s
// MatrixMul5/512/128/1024      5381612 ns      5380919 ns          125 items_per_second=185.842/s
// MatrixMul5/512/512/8          586731 ns       586647 ns         1166 items_per_second=1.7046k/s
// MatrixMul5/512/512/32         878827 ns       878657 ns          784 items_per_second=1.1381k/s
// MatrixMul5/512/512/128       2171977 ns      2171685 ns          322 items_per_second=460.472/s
// MatrixMul5/512/512/512       9777739 ns      9776337 ns           68 items_per_second=102.288/s
// MatrixMul5/512/512/1024     20965703 ns     20962556 ns           33 items_per_second=47.7041/s
// MatrixMul5/512/1024/8        1183673 ns      1183557 ns          588 items_per_second=844.91/s
// MatrixMul5/512/1024/32       1759141 ns      1758816 ns          398 items_per_second=568.564/s
// MatrixMul5/512/1024/128      4490590 ns      4489943 ns          149 items_per_second=222.72/s
// MatrixMul5/512/1024/512     19329550 ns     19326417 ns           36 items_per_second=51.7426/s
// MatrixMul5/512/1024/1024    41278200 ns     41270941 ns           17 items_per_second=24.2301/s
// MatrixMul5/1024/8/8            18763 ns        18758 ns        32534 items_per_second=53.3101k/s
// MatrixMul5/1024/8/32           26424 ns        26421 ns        26571 items_per_second=37.8486k/s
// MatrixMul5/1024/8/128          72390 ns        72380 ns         9650 items_per_second=13.816k/s
// MatrixMul5/1024/8/512         303998 ns       303962 ns         2308 items_per_second=3.28988k/s
// MatrixMul5/1024/8/1024       1799830 ns      1799383 ns          389 items_per_second=555.746/s
// MatrixMul5/1024/32/8           74820 ns        74813 ns         9359 items_per_second=13.3667k/s
// MatrixMul5/1024/32/32         113306 ns       113287 ns         6175 items_per_second=8.82713k/s
// MatrixMul5/1024/32/128        291083 ns       291040 ns         2405 items_per_second=3.43596k/s
// MatrixMul5/1024/32/512       1223002 ns      1222819 ns          568 items_per_second=817.782/s
// MatrixMul5/1024/32/1024      7280639 ns      7278638 ns           81 items_per_second=137.388/s
// MatrixMul5/1024/128/8         302026 ns       301986 ns         2317 items_per_second=3.31141k/s
// MatrixMul5/1024/128/32        455576 ns       455530 ns         1537 items_per_second=2.19524k/s
// MatrixMul5/1024/128/128      1150745 ns      1150630 ns          608 items_per_second=869.089/s
// MatrixMul5/1024/128/512      4907808 ns      4907351 ns          143 items_per_second=203.776/s
// MatrixMul5/1024/128/1024    29326907 ns     29319127 ns           24 items_per_second=34.1074/s
// MatrixMul5/1024/512/8        1208720 ns      1208560 ns          579 items_per_second=827.431/s
// MatrixMul5/1024/512/32       1751039 ns      1750863 ns          400 items_per_second=571.147/s
// MatrixMul5/1024/512/128      4630947 ns      4630605 ns          152 items_per_second=215.955/s
// MatrixMul5/1024/512/512     20470410 ns     20467976 ns           33 items_per_second=48.8568/s
// MatrixMul5/1024/512/1024   110744026 ns    110703297 ns            6 items_per_second=9.03315/s
// MatrixMul5/1024/1024/8       2420600 ns      2419902 ns          289 items_per_second=413.24/s
// MatrixMul5/1024/1024/32      3564556 ns      3564079 ns          198 items_per_second=280.577/s
// MatrixMul5/1024/1024/128     9819547 ns      9816448 ns           75 items_per_second=101.87/s
// MatrixMul5/1024/1024/512    53815471 ns     53812391 ns           10 items_per_second=18.5831/s
// MatrixMul5/1024/1024/1024  669481753 ns    669445965 ns            1 items_per_second=1.49377/s
// MatrixMul6/8/8/8                 283 ns          283 ns      2441187 items_per_second=3.53033M/s
// MatrixMul6/8/8/32                559 ns          559 ns      1260111 items_per_second=1.78875M/s
// MatrixMul6/8/8/128              2239 ns         2239 ns       318101 items_per_second=446.672k/s
// MatrixMul6/8/8/512              8298 ns         8298 ns        85601 items_per_second=120.516k/s
// MatrixMul6/8/8/1024            18795 ns        18793 ns        37236 items_per_second=53.2115k/s
// MatrixMul6/8/32/8               1035 ns         1035 ns       670784 items_per_second=965.842k/s
// MatrixMul6/8/32/32              2125 ns         2125 ns       329719 items_per_second=470.602k/s
// MatrixMul6/8/32/128             8107 ns         8105 ns        85991 items_per_second=123.381k/s
// MatrixMul6/8/32/512            32108 ns        32102 ns        21749 items_per_second=31.1505k/s
// MatrixMul6/8/32/1024           75030 ns        75010 ns         9398 items_per_second=13.3316k/s
// MatrixMul6/8/128/8              4101 ns         4100 ns       170898 items_per_second=243.91k/s
// MatrixMul6/8/128/32             8478 ns         8476 ns        82380 items_per_second=117.975k/s
// MatrixMul6/8/128/128           32369 ns        32365 ns        21434 items_per_second=30.8975k/s
// MatrixMul6/8/128/512          128397 ns       128368 ns         5458 items_per_second=7.79012k/s
// MatrixMul6/8/128/1024         303345 ns       303310 ns         2326 items_per_second=3.29696k/s
// MatrixMul6/8/512/8             16366 ns        16363 ns        42816 items_per_second=61.113k/s
// MatrixMul6/8/512/32            33880 ns        33875 ns        20675 items_per_second=29.52k/s
// MatrixMul6/8/512/128          129500 ns       129476 ns         5406 items_per_second=7.72341k/s
// MatrixMul6/8/512/512          514596 ns       514512 ns         1361 items_per_second=1.94359k/s
// MatrixMul6/8/512/1024        1211466 ns      1211268 ns          577 items_per_second=825.581/s
// MatrixMul6/8/1024/8            32690 ns        32685 ns        21158 items_per_second=30.5954k/s
// MatrixMul6/8/1024/32           67692 ns        67680 ns        10328 items_per_second=14.7755k/s
// MatrixMul6/8/1024/128         258332 ns       258286 ns         2710 items_per_second=3.87167k/s
// MatrixMul6/8/1024/512        1022281 ns      1022170 ns          683 items_per_second=978.311/s
// MatrixMul6/8/1024/1024       2443449 ns      2442924 ns          286 items_per_second=409.346/s
// MatrixMul6/32/8/8               1004 ns         1004 ns       692747 items_per_second=996.446k/s
// MatrixMul6/32/8/32              2056 ns         2056 ns       340407 items_per_second=486.331k/s
// MatrixMul6/32/8/128             7982 ns         7980 ns        86736 items_per_second=125.308k/s
// MatrixMul6/32/8/512            78126 ns        78117 ns         8913 items_per_second=12.8013k/s
// MatrixMul6/32/8/1024          155948 ns       155925 ns         4485 items_per_second=6.41333k/s
// MatrixMul6/32/32/8              3980 ns         3980 ns       175515 items_per_second=251.254k/s
// MatrixMul6/32/32/32             8232 ns         8231 ns        84862 items_per_second=121.499k/s
// MatrixMul6/32/32/128           31902 ns        31900 ns        21936 items_per_second=31.3482k/s
// MatrixMul6/32/32/512          312866 ns       312818 ns         2238 items_per_second=3.19675k/s
// MatrixMul6/32/32/1024         625618 ns       625528 ns         1118 items_per_second=1.59865k/s
// MatrixMul6/32/128/8            16092 ns        16090 ns        43519 items_per_second=62.1522k/s
// MatrixMul6/32/128/32           33023 ns        33019 ns        21219 items_per_second=30.2857k/s
// MatrixMul6/32/128/128         128529 ns       128511 ns         5433 items_per_second=7.78146k/s
// MatrixMul6/32/128/512        1257375 ns      1257160 ns          556 items_per_second=795.444/s
// MatrixMul6/32/128/1024       2510595 ns      2510192 ns          279 items_per_second=398.376/s
// MatrixMul6/32/512/8            67187 ns        67179 ns        10743 items_per_second=14.8856k/s
// MatrixMul6/32/512/32          132034 ns       132012 ns         5301 items_per_second=7.57509k/s
// MatrixMul6/32/512/128         513451 ns       513331 ns         1344 items_per_second=1.94806k/s
// MatrixMul6/32/512/512        5026424 ns      5025600 ns          137 items_per_second=198.981/s
// MatrixMul6/32/512/1024      10045230 ns     10044290 ns           69 items_per_second=99.559/s
// MatrixMul6/32/1024/8          128647 ns       128631 ns         5445 items_per_second=7.7742k/s
// MatrixMul6/32/1024/32         263930 ns       263892 ns         2652 items_per_second=3.78942k/s
// MatrixMul6/32/1024/128       1027267 ns      1027100 ns          680 items_per_second=973.615/s
// MatrixMul6/32/1024/512      10064114 ns     10062173 ns           69 items_per_second=99.3821/s
// MatrixMul6/32/1024/1024     20316884 ns     20312279 ns           34 items_per_second=49.2313/s
// MatrixMul6/128/8/8              3808 ns         3808 ns       183628 items_per_second=262.625k/s
// MatrixMul6/128/8/32             8000 ns         7998 ns        87204 items_per_second=125.025k/s
// MatrixMul6/128/8/128          131620 ns       131609 ns         5356 items_per_second=7.59824k/s
// MatrixMul6/128/8/512          603730 ns       603669 ns         1184 items_per_second=1.65654k/s
// MatrixMul6/128/8/1024        1211578 ns      1211405 ns          575 items_per_second=825.488/s
// MatrixMul6/128/32/8            15132 ns        15129 ns        46188 items_per_second=66.0962k/s
// MatrixMul6/128/32/32           31913 ns        31908 ns        21927 items_per_second=31.3396k/s
// MatrixMul6/128/32/128         521377 ns       521336 ns         1302 items_per_second=1.91815k/s
// MatrixMul6/128/32/512        2325732 ns      2325361 ns          296 items_per_second=430.041/s
// MatrixMul6/128/32/1024       4855223 ns      4854458 ns          145 items_per_second=205.996/s
// MatrixMul6/128/128/8           60806 ns        60798 ns        11443 items_per_second=16.448k/s
// MatrixMul6/128/128/32         128595 ns       128571 ns         5438 items_per_second=7.7778k/s
// MatrixMul6/128/128/128       2141194 ns      2140669 ns          325 items_per_second=467.144/s
// MatrixMul6/128/128/512       9371664 ns      9370472 ns           75 items_per_second=106.718/s
// MatrixMul6/128/128/1024     19346189 ns     19344479 ns           36 items_per_second=51.6943/s
// MatrixMul6/128/512/8          241734 ns       241704 ns         2877 items_per_second=4.1373k/s
// MatrixMul6/128/512/32         510840 ns       510768 ns         1362 items_per_second=1.95783k/s
// MatrixMul6/128/512/128       8438595 ns      8437146 ns           82 items_per_second=118.523/s
// MatrixMul6/128/512/512      37034004 ns     37030252 ns           19 items_per_second=27.0049/s
// MatrixMul6/128/512/1024     77328693 ns     77316035 ns            9 items_per_second=12.9339/s
// MatrixMul6/128/1024/8         483856 ns       483817 ns         1448 items_per_second=2.0669k/s
// MatrixMul6/128/1024/32       1021944 ns      1021782 ns          683 items_per_second=978.682/s
// MatrixMul6/128/1024/128     16908691 ns     16907188 ns           41 items_per_second=59.1464/s
// MatrixMul6/128/1024/512     73918192 ns     73909898 ns            9 items_per_second=13.53/s
// MatrixMul6/128/1024/1024   155876352 ns    155863959 ns            4 items_per_second=6.41585/s
// MatrixMul6/512/8/8             14896 ns        14893 ns        46951 items_per_second=67.1461k/s
// MatrixMul6/512/8/32           428341 ns       428271 ns         1629 items_per_second=2.33497k/s
// MatrixMul6/512/8/128         3084938 ns      3084584 ns          228 items_per_second=324.193/s
// MatrixMul6/512/8/512        12819538 ns     12817552 ns           55 items_per_second=78.018/s
// MatrixMul6/512/8/1024       25777686 ns     25770804 ns           27 items_per_second=38.8036/s
// MatrixMul6/512/32/8            60025 ns        60010 ns        11446 items_per_second=16.664k/s
// MatrixMul6/512/32/32         1796286 ns      1796051 ns          393 items_per_second=556.777/s
// MatrixMul6/512/32/128       10910555 ns     10908349 ns           66 items_per_second=91.6729/s
// MatrixMul6/512/32/512       51682654 ns     51675808 ns           14 items_per_second=19.3514/s
// MatrixMul6/512/32/1024     103813354 ns    103803397 ns            7 items_per_second=9.6336/s
// MatrixMul6/512/128/8          240295 ns       240270 ns         2851 items_per_second=4.16198k/s
// MatrixMul6/512/128/32        7233370 ns      7232734 ns           96 items_per_second=138.26/s
// MatrixMul6/512/128/128      45181074 ns     45172722 ns           15 items_per_second=22.1373/s
// MatrixMul6/512/128/512     199512028 ns    199481254 ns            3 items_per_second=5.013/s
// MatrixMul6/512/128/1024    413571030 ns    413473794 ns            2 items_per_second=2.41853/s
// MatrixMul6/512/512/8          954609 ns       954486 ns          727 items_per_second=1.04768k/s
// MatrixMul6/512/512/32       28322153 ns     28316306 ns           25 items_per_second=35.3153/s
// MatrixMul6/512/512/128     175938104 ns    175922336 ns            4 items_per_second=5.68433/s
// MatrixMul6/512/512/512     793544851 ns    793411471 ns            1 items_per_second=1.26038/s
// MatrixMul6/512/512/1024   1603691692 ns   1603472516 ns            1 items_per_second=0.623646/s
// MatrixMul6/512/1024/8        1954738 ns      1954685 ns          339 items_per_second=511.591/s
// MatrixMul6/512/1024/32      57257696 ns     57251822 ns           12 items_per_second=17.4667/s
// MatrixMul6/512/1024/128    367404657 ns    367377289 ns            2 items_per_second=2.722/s
// MatrixMul6/512/1024/512   1653111375 ns   1652883401 ns            1 items_per_second=0.605003/s
// MatrixMul6/512/1024/1024  3286716315 ns   3286152084 ns            1 items_per_second=0.304307/s
// MatrixMul6/1024/8/8            35286 ns        35282 ns        20806 items_per_second=28.3431k/s
// MatrixMul6/1024/8/32         1728803 ns      1728580 ns          404 items_per_second=578.51/s
// MatrixMul6/1024/8/128        9465942 ns      9464096 ns           74 items_per_second=105.662/s
// MatrixMul6/1024/8/512       39500686 ns     39490959 ns           18 items_per_second=25.3223/s
// MatrixMul6/1024/8/1024      83656272 ns     83633687 ns            8 items_per_second=11.9569/s
// MatrixMul6/1024/32/8          130751 ns       130726 ns         5349 items_per_second=7.6496k/s
// MatrixMul6/1024/32/32        6938182 ns      6937693 ns          101 items_per_second=144.14/s
// MatrixMul6/1024/32/128      32848413 ns     32844211 ns           21 items_per_second=30.4468/s
// MatrixMul6/1024/32/512     156189807 ns    156170724 ns            5 items_per_second=6.40325/s
// MatrixMul6/1024/32/1024    333369480 ns    333323096 ns            2 items_per_second=3.00009/s
// MatrixMul6/1024/128/8        1269264 ns      1269067 ns          549 items_per_second=787.98/s
// MatrixMul6/1024/128/32      27829940 ns     27826849 ns           25 items_per_second=35.9365/s
// MatrixMul6/1024/128/128    139910684 ns    139892759 ns            5 items_per_second=7.14833/s
// MatrixMul6/1024/128/512    628313361 ns    628195747 ns            1 items_per_second=1.59186/s
// MatrixMul6/1024/128/1024  1337588155 ns   1337335638 ns            1 items_per_second=0.747755/s
// MatrixMul6/1024/512/8        5105160 ns      5103866 ns          136 items_per_second=195.93/s
// MatrixMul6/1024/512/32     111362583 ns    111354538 ns            6 items_per_second=8.98033/s
// MatrixMul6/1024/512/128    588367719 ns    588299714 ns            1 items_per_second=1.69981/s
// MatrixMul6/1024/512/512   2491247042 ns   2490958205 ns            1 items_per_second=0.401452/s
// MatrixMul6/1024/512/1024  5313878326 ns   5312819040 ns            1 items_per_second=0.188224/s
// MatrixMul6/1024/1024/8      10220340 ns     10218329 ns           68 items_per_second=97.8634/s
// MatrixMul6/1024/1024/32    222375133 ns    222355281 ns            3 items_per_second=4.49731/s
// MatrixMul6/1024/1024/128  1121194305 ns   1121035321 ns            1 items_per_second=0.892033/s
// MatrixMul6/1024/1024/512  4971657461 ns   4971027484 ns            1 items_per_second=0.201166/s
// MatrixMul6/1024/1024/1024 1.0858e+10 ns   1.0856e+10 ns            1 items_per_second=0.0921179/s
// MatrixMul7/8/8/8                 176 ns          176 ns      3969241 items_per_second=5.67466M/s
// MatrixMul7/8/8/32                224 ns          224 ns      3115514 items_per_second=4.45688M/s
// MatrixMul7/8/8/128               413 ns          413 ns      1693591 items_per_second=2.41996M/s
// MatrixMul7/8/8/512              1377 ns         1377 ns       509714 items_per_second=726.409k/s
// MatrixMul7/8/8/1024             3542 ns         3541 ns       196844 items_per_second=282.378k/s
// MatrixMul7/8/32/8                682 ns          682 ns      1021984 items_per_second=1.46621M/s
// MatrixMul7/8/32/32               876 ns          876 ns       796295 items_per_second=1.14117M/s
// MatrixMul7/8/32/128             1628 ns         1628 ns       429398 items_per_second=614.182k/s
// MatrixMul7/8/32/512             8518 ns         8516 ns        81930 items_per_second=117.423k/s
// MatrixMul7/8/32/1024           17343 ns        17340 ns        40332 items_per_second=57.6687k/s
// MatrixMul7/8/128/8              2713 ns         2712 ns       258244 items_per_second=368.694k/s
// MatrixMul7/8/128/32             3467 ns         3466 ns       202073 items_per_second=288.485k/s
// MatrixMul7/8/128/128            6791 ns         6789 ns       102691 items_per_second=147.286k/s
// MatrixMul7/8/128/512           18725 ns        18723 ns        37258 items_per_second=53.4104k/s
// MatrixMul7/8/128/1024          82380 ns        82369 ns         8484 items_per_second=12.1405k/s
// MatrixMul7/8/512/8             10779 ns        10778 ns        64880 items_per_second=92.7841k/s
// MatrixMul7/8/512/32            13774 ns        13773 ns        50822 items_per_second=72.6077k/s
// MatrixMul7/8/512/128           27080 ns        27074 ns        25859 items_per_second=36.9361k/s
// MatrixMul7/8/512/512           77426 ns        77415 ns         9058 items_per_second=12.9174k/s
// MatrixMul7/8/512/1024         348209 ns       348146 ns         2009 items_per_second=2.87236k/s
// MatrixMul7/8/1024/8            21554 ns        21551 ns        32467 items_per_second=46.4021k/s
// MatrixMul7/8/1024/32           27555 ns        27548 ns        25411 items_per_second=36.3008k/s
// MatrixMul7/8/1024/128          71953 ns        71946 ns         9707 items_per_second=13.8994k/s
// MatrixMul7/8/1024/512         331676 ns       331600 ns         2109 items_per_second=3.01568k/s
// MatrixMul7/8/1024/1024        789099 ns       788955 ns          877 items_per_second=1.2675k/s
// MatrixMul7/32/8/8                695 ns          695 ns      1003858 items_per_second=1.43951M/s
// MatrixMul7/32/8/32               877 ns          877 ns       799443 items_per_second=1.13985M/s
// MatrixMul7/32/8/128             1619 ns         1619 ns       433755 items_per_second=617.792k/s
// MatrixMul7/32/8/512             7661 ns         7659 ns        91075 items_per_second=130.557k/s
// MatrixMul7/32/8/1024           15750 ns        15747 ns        44499 items_per_second=63.5024k/s
// MatrixMul7/32/32/8              2773 ns         2773 ns       253890 items_per_second=360.654k/s
// MatrixMul7/32/32/32             3499 ns         3498 ns       200851 items_per_second=285.88k/s
// MatrixMul7/32/32/128            6320 ns         6319 ns       108224 items_per_second=158.246k/s
// MatrixMul7/32/32/512           36808 ns        36803 ns        18908 items_per_second=27.1716k/s
// MatrixMul7/32/32/1024          74880 ns        74872 ns         9390 items_per_second=13.3561k/s
// MatrixMul7/32/128/8            11694 ns        11693 ns        59844 items_per_second=85.524k/s
// MatrixMul7/32/128/32           13928 ns        13926 ns        50438 items_per_second=71.8091k/s
// MatrixMul7/32/128/128          25527 ns        25523 ns        27409 items_per_second=39.1809k/s
// MatrixMul7/32/128/512         146282 ns       146260 ns         4762 items_per_second=6.83714k/s
// MatrixMul7/32/128/1024        300635 ns       300577 ns         2329 items_per_second=3.32694k/s
// MatrixMul7/32/512/8            44190 ns        44186 ns        14935 items_per_second=22.6318k/s
// MatrixMul7/32/512/32           55827 ns        55819 ns        12401 items_per_second=17.9149k/s
// MatrixMul7/32/512/128         136579 ns       136558 ns         5055 items_per_second=7.32291k/s
// MatrixMul7/32/512/512         526964 ns       526859 ns         1325 items_per_second=1.89804k/s
// MatrixMul7/32/512/1024       1102067 ns      1101907 ns          634 items_per_second=907.517/s
// MatrixMul7/32/1024/8           87360 ns        87348 ns         7905 items_per_second=11.4484k/s
// MatrixMul7/32/1024/32         110983 ns       110959 ns         6305 items_per_second=9.01232k/s
// MatrixMul7/32/1024/128        272701 ns       272666 ns         2565 items_per_second=3.6675k/s
// MatrixMul7/32/1024/512       1054229 ns      1054031 ns          664 items_per_second=948.738/s
// MatrixMul7/32/1024/1024      2334048 ns      2333614 ns          301 items_per_second=428.52/s
// MatrixMul7/128/8/8              2561 ns         2560 ns       273929 items_per_second=390.598k/s
// MatrixMul7/128/8/32             3307 ns         3306 ns       211502 items_per_second=302.459k/s
// MatrixMul7/128/8/128            6910 ns         6909 ns        99418 items_per_second=144.728k/s
// MatrixMul7/128/8/512           26825 ns        26823 ns        26103 items_per_second=37.2817k/s
// MatrixMul7/128/8/1024          75902 ns        75893 ns         9089 items_per_second=13.1764k/s
// MatrixMul7/128/32/8            11166 ns        11165 ns        61599 items_per_second=89.5658k/s
// MatrixMul7/128/32/32           13239 ns        13238 ns        52736 items_per_second=75.5428k/s
// MatrixMul7/128/32/128          27530 ns        27525 ns        25355 items_per_second=36.3312k/s
// MatrixMul7/128/32/512         143404 ns       143383 ns         4903 items_per_second=6.97435k/s
// MatrixMul7/128/32/1024        306837 ns       306790 ns         2243 items_per_second=3.25956k/s
// MatrixMul7/128/128/8           40799 ns        40794 ns        17164 items_per_second=24.5134k/s
// MatrixMul7/128/128/32          54920 ns        54903 ns        12843 items_per_second=18.2139k/s
// MatrixMul7/128/128/128        140902 ns       140880 ns         4835 items_per_second=7.09822k/s
// MatrixMul7/128/128/512        537710 ns       537590 ns         1273 items_per_second=1.86015k/s
// MatrixMul7/128/128/1024      1213118 ns      1212971 ns          576 items_per_second=824.422/s
// MatrixMul7/128/512/8          163438 ns       163412 ns         4275 items_per_second=6.11949k/s
// MatrixMul7/128/512/32         214752 ns       214706 ns         3259 items_per_second=4.65753k/s
// MatrixMul7/128/512/128        540804 ns       540706 ns         1280 items_per_second=1.84943k/s
// MatrixMul7/128/512/512       2164686 ns      2164293 ns          317 items_per_second=462.045/s
// MatrixMul7/128/512/1024      4815421 ns      4814884 ns          145 items_per_second=207.689/s
// MatrixMul7/128/1024/8         328184 ns       328134 ns         2123 items_per_second=3.04753k/s
// MatrixMul7/128/1024/32        428701 ns       428664 ns         1629 items_per_second=2.33283k/s
// MatrixMul7/128/1024/128      1085020 ns      1084827 ns          642 items_per_second=921.806/s
// MatrixMul7/128/1024/512      4319791 ns      4319180 ns          162 items_per_second=231.525/s
// MatrixMul7/128/1024/1024    10025738 ns     10023410 ns           70 items_per_second=99.7664/s
// MatrixMul7/512/8/8             10016 ns        10015 ns        69874 items_per_second=99.8525k/s
// MatrixMul7/512/8/32            13539 ns        13536 ns        51694 items_per_second=73.8754k/s
// MatrixMul7/512/8/128           28315 ns        28312 ns        24760 items_per_second=35.3212k/s
// MatrixMul7/512/8/512          131344 ns       131324 ns         5294 items_per_second=7.61473k/s
// MatrixMul7/512/8/1024         313257 ns       313207 ns         2213 items_per_second=3.19278k/s
// MatrixMul7/512/32/8            39979 ns        39973 ns        17501 items_per_second=25.0172k/s
// MatrixMul7/512/32/32           56760 ns        56753 ns        12312 items_per_second=17.6203k/s
// MatrixMul7/512/32/128         141245 ns       141221 ns         4951 items_per_second=7.08108k/s
// MatrixMul7/512/32/512         602651 ns       602563 ns         1154 items_per_second=1.65958k/s
// MatrixMul7/512/32/1024       1277775 ns      1277623 ns          546 items_per_second=782.704/s
// MatrixMul7/512/128/8          160050 ns       160022 ns         4366 items_per_second=6.24915k/s
// MatrixMul7/512/128/32         227216 ns       227185 ns         3082 items_per_second=4.4017k/s
// MatrixMul7/512/128/128        547059 ns       546944 ns         1274 items_per_second=1.82834k/s
// MatrixMul7/512/128/512       2519366 ns      2519102 ns          278 items_per_second=396.967/s
// MatrixMul7/512/128/1024      5113131 ns      5112190 ns          131 items_per_second=195.611/s
// MatrixMul7/512/512/8          641907 ns       641835 ns         1085 items_per_second=1.55803k/s
// MatrixMul7/512/512/32         889739 ns       889639 ns          786 items_per_second=1.12405k/s
// MatrixMul7/512/512/128       2178203 ns      2177690 ns          322 items_per_second=459.202/s
// MatrixMul7/512/512/512      10030227 ns     10029070 ns           69 items_per_second=99.7101/s
// MatrixMul7/512/512/1024     20849021 ns     20845768 ns           34 items_per_second=47.9714/s
// MatrixMul7/512/1024/8        1291186 ns      1290989 ns          541 items_per_second=774.6/s
// MatrixMul7/512/1024/32       1774573 ns      1774288 ns          394 items_per_second=563.606/s
// MatrixMul7/512/1024/128      4343591 ns      4343141 ns          156 items_per_second=230.248/s
// MatrixMul7/512/1024/512     20056066 ns     20052044 ns           36 items_per_second=49.8702/s
// MatrixMul7/512/1024/1024    42512244 ns     42505335 ns           17 items_per_second=23.5265/s
// MatrixMul7/1024/8/8            20078 ns        20074 ns        34898 items_per_second=49.8154k/s
// MatrixMul7/1024/8/32           26959 ns        26956 ns        25984 items_per_second=37.0981k/s
// MatrixMul7/1024/8/128          71484 ns        71471 ns         9778 items_per_second=13.9917k/s
// MatrixMul7/1024/8/512         299152 ns       299122 ns         2335 items_per_second=3.34312k/s
// MatrixMul7/1024/8/1024       1808879 ns      1808373 ns          370 items_per_second=552.983/s
// MatrixMul7/1024/32/8           80336 ns        80319 ns         8718 items_per_second=12.4504k/s
// MatrixMul7/1024/32/32         114146 ns       114133 ns         6133 items_per_second=8.76174k/s
// MatrixMul7/1024/32/128        301509 ns       301484 ns         2331 items_per_second=3.31692k/s
// MatrixMul7/1024/32/512       1217778 ns      1217667 ns          574 items_per_second=821.243/s
// MatrixMul7/1024/32/1024      7156237 ns      7154267 ns           88 items_per_second=139.777/s
// MatrixMul7/1024/128/8         322367 ns       322337 ns         2174 items_per_second=3.10235k/s
// MatrixMul7/1024/128/32        459136 ns       459032 ns         1528 items_per_second=2.1785k/s
// MatrixMul7/1024/128/128      1187363 ns      1187208 ns          606 items_per_second=842.312/s
// MatrixMul7/1024/128/512      5018009 ns      5017391 ns          143 items_per_second=199.307/s
// MatrixMul7/1024/128/1024    27190558 ns     27186511 ns           26 items_per_second=36.7829/s
// MatrixMul7/1024/512/8        1289169 ns      1288873 ns          543 items_per_second=775.872/s
// MatrixMul7/1024/512/32       1778641 ns      1778043 ns          394 items_per_second=562.416/s
// MatrixMul7/1024/512/128      4567720 ns      4566850 ns          153 items_per_second=218.969/s
// MatrixMul7/1024/512/512     20510461 ns     20507506 ns           35 items_per_second=48.7626/s
// MatrixMul7/1024/512/1024   113268789 ns    113216876 ns            6 items_per_second=8.83261/s
// MatrixMul7/1024/1024/8       2584493 ns      2584093 ns          271 items_per_second=386.983/s
// MatrixMul7/1024/1024/32      3544465 ns      3543486 ns          197 items_per_second=282.208/s
// MatrixMul7/1024/1024/128     9176555 ns      9174597 ns           76 items_per_second=108.997/s
// MatrixMul7/1024/1024/512    39579849 ns     39568262 ns           18 items_per_second=25.2728/s
// MatrixMul7/1024/1024/1024  227550734 ns    227482267 ns            3 items_per_second=4.39595/s