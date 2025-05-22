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
BENCHMARK(MatrixMul7)->Apply(CustomArgs);

BENCHMARK_MAIN();
// ---------------------------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------
// MatrixMul1/1/1/1             21.8 ns         21.8 ns     32821506 items_per_second=45.969M/s
// MatrixMul1/1/2/1             22.1 ns         22.1 ns     32609074 items_per_second=45.348M/s
// MatrixMul1/1/4/1             22.2 ns         22.2 ns     31571585 items_per_second=44.9562M/s
// MatrixMul1/1/6/1             22.4 ns         22.4 ns     30961169 items_per_second=44.5609M/s
// MatrixMul1/1/8/1             22.3 ns         22.3 ns     31463532 items_per_second=44.8021M/s
// MatrixMul1/1/10/1            22.8 ns         22.8 ns     31027025 items_per_second=43.8663M/s
// MatrixMul1/1/16/1            25.0 ns         25.0 ns     27925909 items_per_second=39.9618M/s
// MatrixMul1/1/32/1            37.1 ns         37.0 ns     18935256 items_per_second=26.9932M/s
// MatrixMul1/1/64/1            60.2 ns         60.2 ns     11577945 items_per_second=16.6202M/s
// MatrixMul1/1/256/1            200 ns          200 ns      3488978 items_per_second=4.98918M/s
// MatrixMul1/1/512/1            388 ns          388 ns      1804476 items_per_second=2.57886M/s
// MatrixMul1/1/1024/1           763 ns          763 ns       916227 items_per_second=1.31133M/s
// MatrixMul1/1/2048/1          1512 ns         1512 ns       462941 items_per_second=661.578k/s
// MatrixMul1/128/1/1            372 ns          372 ns      1847156 items_per_second=2.68569M/s
// MatrixMul1/128/2/1            436 ns          436 ns      1615250 items_per_second=2.29298M/s
// MatrixMul1/128/4/1            526 ns          526 ns      1292240 items_per_second=1.89962M/s
// MatrixMul1/128/6/1            623 ns          622 ns      1110602 items_per_second=1.60644M/s
// MatrixMul1/128/8/1            608 ns          607 ns      1118014 items_per_second=1.64631M/s
// MatrixMul1/128/10/1           718 ns          718 ns       952943 items_per_second=1.39345M/s
// MatrixMul1/128/16/1          1011 ns         1011 ns       697798 items_per_second=988.914k/s
// MatrixMul1/128/32/1          2101 ns         2101 ns       331651 items_per_second=475.973k/s
// MatrixMul1/128/64/1          4721 ns         4720 ns       147869 items_per_second=211.856k/s
// MatrixMul1/128/256/1        22814 ns        22811 ns        30576 items_per_second=43.8386k/s
// MatrixMul1/128/512/1        47241 ns        47235 ns        14897 items_per_second=21.1707k/s
// MatrixMul1/128/1024/1       96319 ns        96311 ns         7365 items_per_second=10.3831k/s
// MatrixMul1/128/2048/1      192166 ns       192107 ns         3618 items_per_second=5.20544k/s
// MatrixMul1/512/1/1           1508 ns         1507 ns       482448 items_per_second=663.407k/s
// MatrixMul1/512/2/1           1701 ns         1701 ns       412622 items_per_second=588.016k/s
// MatrixMul1/512/4/1           2064 ns         2064 ns       339293 items_per_second=484.566k/s
// MatrixMul1/512/6/1           2453 ns         2453 ns       285365 items_per_second=407.708k/s
// MatrixMul1/512/8/1           2768 ns         2768 ns       288933 items_per_second=361.287k/s
// MatrixMul1/512/10/1          2976 ns         2976 ns       234542 items_per_second=336.014k/s
// MatrixMul1/512/16/1          3971 ns         3971 ns       175101 items_per_second=251.837k/s
// MatrixMul1/512/32/1          8259 ns         8257 ns        85095 items_per_second=121.111k/s
// MatrixMul1/512/64/1         19035 ns        19034 ns        36503 items_per_second=52.5372k/s
// MatrixMul1/512/256/1        90741 ns        90722 ns         7675 items_per_second=11.0227k/s
// MatrixMul1/512/512/1       186794 ns       186766 ns         3744 items_per_second=5.35429k/s
// MatrixMul1/512/1024/1      394666 ns       394570 ns         1820 items_per_second=2.5344k/s
// MatrixMul1/512/2048/1      783159 ns       782983 ns          895 items_per_second=1.27717k/s
// MatrixMul1/64/256/32       362934 ns       362869 ns         1928 items_per_second=2.75581k/s
// MatrixMul1/64/256/64       760380 ns       760255 ns          918 items_per_second=1.31535k/s
// MatrixMul1/64/256/128     2013982 ns      2013762 ns          348 items_per_second=496.583/s
// MatrixMul1/64/256/512    17992679 ns     17989538 ns           39 items_per_second=55.5879/s
// MatrixMul1/128/256/32      744332 ns       744256 ns          940 items_per_second=1.34362k/s
// MatrixMul1/128/256/64     1457555 ns      1457314 ns          482 items_per_second=686.194/s
// MatrixMul1/128/256/128    3953602 ns      3953005 ns          176 items_per_second=252.972/s
// MatrixMul1/128/256/512   34277448 ns     34273521 ns           20 items_per_second=29.177/s
// MatrixMul1/256/256/32     1487039 ns      1486805 ns          473 items_per_second=672.583/s
// MatrixMul1/256/256/64     2923471 ns      2923318 ns          238 items_per_second=342.077/s
// MatrixMul1/256/256/128    7923415 ns      7922175 ns           87 items_per_second=126.228/s
// MatrixMul1/256/256/512   72146780 ns     72138034 ns           10 items_per_second=13.8623/s
// MatrixMul1/512/256/32     2909513 ns      2909063 ns          240 items_per_second=343.753/s
// MatrixMul1/512/256/64     5867268 ns      5866434 ns          118 items_per_second=170.461/s
// MatrixMul1/512/256/128   15787871 ns     15785605 ns           44 items_per_second=63.3489/s
// MatrixMul1/512/256/512  143809319 ns    143785949 ns            5 items_per_second=6.95478/s
// MatrixMul1/64/512/32       749275 ns       749117 ns          932 items_per_second=1.3349k/s
// MatrixMul1/64/512/64      1541914 ns      1541706 ns          456 items_per_second=648.632/s
// MatrixMul1/64/512/128     3951311 ns      3950908 ns          177 items_per_second=253.106/s
// MatrixMul1/64/512/512    35243426 ns     35239258 ns           20 items_per_second=28.3774/s
// MatrixMul1/128/512/32     1518467 ns      1518165 ns          461 items_per_second=658.69/s
// MatrixMul1/128/512/64     3001995 ns      3001706 ns          233 items_per_second=333.144/s
// MatrixMul1/128/512/128    7972607 ns      7971661 ns           88 items_per_second=125.444/s
// MatrixMul1/128/512/512   69995373 ns     69983014 ns           10 items_per_second=14.2892/s
// MatrixMul1/256/512/32     3005805 ns      3005373 ns          233 items_per_second=332.737/s
// MatrixMul1/256/512/64     5993642 ns      5992819 ns          116 items_per_second=166.866/s
// MatrixMul1/256/512/128   15807724 ns     15805965 ns           43 items_per_second=63.2673/s
// MatrixMul1/256/512/512  141202243 ns    141181172 ns            5 items_per_second=7.0831/s
// MatrixMul1/512/512/32     5999843 ns      5999004 ns          116 items_per_second=166.694/s
// MatrixMul1/512/512/64    11989002 ns     11986248 ns           58 items_per_second=83.4289/s
// MatrixMul1/512/512/128   32639976 ns     32634419 ns           21 items_per_second=30.6425/s
// MatrixMul1/512/512/512  283639180 ns    283599186 ns            2 items_per_second=3.5261/s
// MatrixMul2/1/1/1             5.36 ns         5.36 ns    134437805 items_per_second=186.614M/s
// MatrixMul2/1/2/1             10.7 ns         10.7 ns     65197028 items_per_second=93.1518M/s
// MatrixMul2/1/4/1             20.7 ns         20.7 ns     33822923 items_per_second=48.2886M/s
// MatrixMul2/1/6/1             29.6 ns         29.6 ns     23640216 items_per_second=33.7453M/s
// MatrixMul2/1/8/1             38.8 ns         38.8 ns     18061604 items_per_second=25.7903M/s
// MatrixMul2/1/10/1            47.9 ns         47.9 ns     12284666 items_per_second=20.8597M/s
// MatrixMul2/1/16/1            81.5 ns         81.5 ns      8580453 items_per_second=12.267M/s
// MatrixMul2/1/32/1             155 ns          155 ns      4525513 items_per_second=6.46458M/s
// MatrixMul2/1/64/1             301 ns          301 ns      2323907 items_per_second=3.32145M/s
// MatrixMul2/1/256/1           1179 ns         1179 ns       593433 items_per_second=848.031k/s
// MatrixMul2/1/512/1           2351 ns         2350 ns       297993 items_per_second=425.463k/s
// MatrixMul2/1/1024/1          4710 ns         4710 ns       149222 items_per_second=212.334k/s
// MatrixMul2/1/2048/1          9374 ns         9373 ns        64133 items_per_second=106.692k/s
// MatrixMul2/128/1/1            501 ns          501 ns      1395560 items_per_second=1.99578M/s
// MatrixMul2/128/2/1           1008 ns         1008 ns       691832 items_per_second=992.381k/s
// MatrixMul2/128/4/1           2006 ns         2006 ns       348643 items_per_second=498.448k/s
// MatrixMul2/128/6/1           3006 ns         3005 ns       232775 items_per_second=332.74k/s
// MatrixMul2/128/8/1           4063 ns         4062 ns       174665 items_per_second=246.177k/s
// MatrixMul2/128/10/1          5067 ns         5067 ns       139347 items_per_second=197.365k/s
// MatrixMul2/128/16/1          8027 ns         8026 ns        87081 items_per_second=124.595k/s
// MatrixMul2/128/32/1         16219 ns        16218 ns        43393 items_per_second=61.6596k/s
// MatrixMul2/128/64/1         32083 ns        32081 ns        21810 items_per_second=31.1709k/s
// MatrixMul2/128/256/1       129045 ns       129031 ns         5442 items_per_second=7.75005k/s
// MatrixMul2/128/512/1       257580 ns       257548 ns         2720 items_per_second=3.88277k/s
// MatrixMul2/128/1024/1      519731 ns       519634 ns         1336 items_per_second=1.92443k/s
// MatrixMul2/128/2048/1     1047447 ns      1047317 ns          671 items_per_second=954.82/s
// MatrixMul2/512/1/1           1985 ns         1985 ns       352130 items_per_second=503.749k/s
// MatrixMul2/512/2/1           3986 ns         3986 ns       176026 items_per_second=250.891k/s
// MatrixMul2/512/4/1           8036 ns         8035 ns        87152 items_per_second=124.463k/s
// MatrixMul2/512/6/1          11923 ns        11922 ns        58810 items_per_second=83.879k/s
// MatrixMul2/512/8/1          15851 ns        15849 ns        44066 items_per_second=63.0936k/s
// MatrixMul2/512/10/1         20687 ns        20686 ns        34890 items_per_second=48.3426k/s
// MatrixMul2/512/16/1         32820 ns        32798 ns        21258 items_per_second=30.4899k/s
// MatrixMul2/512/32/1         64038 ns        64022 ns        10738 items_per_second=15.6197k/s
// MatrixMul2/512/64/1        132619 ns       132599 ns         5415 items_per_second=7.54151k/s
// MatrixMul2/512/256/1       525028 ns       525004 ns         1334 items_per_second=1.90475k/s
// MatrixMul2/512/512/1      1198872 ns      1198746 ns          567 items_per_second=834.205/s
// MatrixMul2/512/1024/1     2125110 ns      2124836 ns          280 items_per_second=470.625/s
// MatrixMul2/512/2048/1     4208334 ns      4207573 ns          165 items_per_second=237.667/s
// MatrixMul2/64/256/32      1853685 ns      1853424 ns          377 items_per_second=539.542/s
// MatrixMul2/64/256/64      3576939 ns      3576392 ns          196 items_per_second=279.611/s
// MatrixMul2/64/256/128     7008695 ns      7008002 ns          100 items_per_second=142.694/s
// MatrixMul2/64/256/512    27640713 ns     27635934 ns           25 items_per_second=36.1848/s
// MatrixMul2/128/256/32     3697126 ns      3696500 ns          189 items_per_second=270.526/s
// MatrixMul2/128/256/64     7127055 ns      7125883 ns           98 items_per_second=140.333/s
// MatrixMul2/128/256/128   14001971 ns     14000424 ns           50 items_per_second=71.4264/s
// MatrixMul2/128/256/512   55236261 ns     55224743 ns           13 items_per_second=18.1078/s
// MatrixMul2/256/256/32     7374684 ns      7373506 ns           95 items_per_second=135.621/s
// MatrixMul2/256/256/64    14270931 ns     14268514 ns           49 items_per_second=70.0844/s
// MatrixMul2/256/256/128   28026086 ns     28020917 ns           25 items_per_second=35.6876/s
// MatrixMul2/256/256/512  110537627 ns    110521510 ns            6 items_per_second=9.04801/s
// MatrixMul2/512/256/32    14776778 ns     14774121 ns           47 items_per_second=67.6859/s
// MatrixMul2/512/256/64    28529946 ns     28525737 ns           25 items_per_second=35.0561/s
// MatrixMul2/512/256/128   55988118 ns     55977980 ns           13 items_per_second=17.8642/s
// MatrixMul2/512/256/512  221652153 ns    221631563 ns            3 items_per_second=4.51199/s
// MatrixMul2/64/512/32      3704690 ns      3704081 ns          190 items_per_second=269.973/s
// MatrixMul2/64/512/64      7129600 ns      7128767 ns           98 items_per_second=140.277/s
// MatrixMul2/64/512/128    14012850 ns     14010652 ns           50 items_per_second=71.3743/s
// MatrixMul2/64/512/512    55114141 ns     55106517 ns           13 items_per_second=18.1467/s
// MatrixMul2/128/512/32     7393044 ns      7392331 ns           95 items_per_second=135.275/s
// MatrixMul2/128/512/64    14258457 ns     14255766 ns           49 items_per_second=70.1471/s
// MatrixMul2/128/512/128   28150674 ns     28148431 ns           25 items_per_second=35.526/s
// MatrixMul2/128/512/512  110457452 ns    110443795 ns            6 items_per_second=9.05438/s
// MatrixMul2/256/512/32    14749042 ns     14746853 ns           47 items_per_second=67.8111/s
// MatrixMul2/256/512/64    28447129 ns     28443213 ns           25 items_per_second=35.1578/s
// MatrixMul2/256/512/128   58162837 ns     58158989 ns           13 items_per_second=17.1942/s
// MatrixMul2/256/512/512  221029541 ns    221015857 ns            3 items_per_second=4.52456/s
// MatrixMul2/512/512/32    29525759 ns     29521339 ns           24 items_per_second=33.8738/s
// MatrixMul2/512/512/64    56938975 ns     56933869 ns           12 items_per_second=17.5642/s
// MatrixMul2/512/512/128  113042091 ns    113026243 ns            6 items_per_second=8.8475/s
// MatrixMul2/512/512/512  442079738 ns    442027101 ns            2 items_per_second=2.2623/s
// MatrixMul3/1/1/1             5.48 ns         5.48 ns    126959995 items_per_second=182.357M/s
// MatrixMul3/1/2/1             9.47 ns         9.47 ns     73916887 items_per_second=105.639M/s
// MatrixMul3/1/4/1             15.9 ns         15.9 ns     43891737 items_per_second=62.8427M/s
// MatrixMul3/1/6/1             23.4 ns         23.4 ns     30068569 items_per_second=42.8M/s
// MatrixMul3/1/8/1             30.5 ns         30.5 ns     23293119 items_per_second=32.832M/s
// MatrixMul3/1/10/1            36.4 ns         36.4 ns     19231461 items_per_second=27.4565M/s
// MatrixMul3/1/16/1            62.1 ns         62.0 ns     11213275 items_per_second=16.1175M/s
// MatrixMul3/1/32/1             114 ns          114 ns      6121255 items_per_second=8.76063M/s
// MatrixMul3/1/64/1             218 ns          218 ns      3211494 items_per_second=4.58672M/s
// MatrixMul3/1/256/1            842 ns          842 ns       828951 items_per_second=1.18808M/s
// MatrixMul3/1/512/1           1673 ns         1673 ns       418118 items_per_second=597.719k/s
// MatrixMul3/1/1024/1          3335 ns         3335 ns       209882 items_per_second=299.881k/s
// MatrixMul3/1/2048/1          6669 ns         6668 ns       104642 items_per_second=149.966k/s
// MatrixMul3/128/1/1            595 ns          595 ns      1173573 items_per_second=1.68072M/s
// MatrixMul3/128/2/1            997 ns          997 ns       698194 items_per_second=1.00345M/s
// MatrixMul3/128/4/1           1832 ns         1832 ns       382536 items_per_second=545.935k/s
// MatrixMul3/128/6/1           2796 ns         2795 ns       250264 items_per_second=357.725k/s
// MatrixMul3/128/8/1           3633 ns         3632 ns       192861 items_per_second=275.318k/s
// MatrixMul3/128/10/1          4453 ns         4453 ns       156971 items_per_second=224.583k/s
// MatrixMul3/128/16/1          7752 ns         7750 ns        89964 items_per_second=129.025k/s
// MatrixMul3/128/32/1         14576 ns        14573 ns        48378 items_per_second=68.622k/s
// MatrixMul3/128/64/1         28100 ns        28096 ns        24961 items_per_second=35.5928k/s
// MatrixMul3/128/256/1       109011 ns       108998 ns         6416 items_per_second=9.17447k/s
// MatrixMul3/128/512/1       216949 ns       216921 ns         3228 items_per_second=4.60997k/s
// MatrixMul3/128/1024/1      431182 ns       431126 ns         1619 items_per_second=2.31951k/s
// MatrixMul3/128/2048/1      861865 ns       861789 ns          810 items_per_second=1.16038k/s
// MatrixMul3/512/1/1           2353 ns         2353 ns       297264 items_per_second=425.013k/s
// MatrixMul3/512/2/1           3955 ns         3954 ns       177052 items_per_second=252.879k/s
// MatrixMul3/512/4/1           7334 ns         7333 ns        95646 items_per_second=136.363k/s
// MatrixMul3/512/6/1          11094 ns        11092 ns        63105 items_per_second=90.1511k/s
// MatrixMul3/512/8/1          14417 ns        14416 ns        48547 items_per_second=69.3688k/s
// MatrixMul3/512/10/1         17773 ns        17770 ns        39413 items_per_second=56.2732k/s
// MatrixMul3/512/16/1         30890 ns        30884 ns        22662 items_per_second=32.3792k/s
// MatrixMul3/512/32/1         57711 ns        57702 ns        12127 items_per_second=17.3304k/s
// MatrixMul3/512/64/1        111332 ns       111323 ns         6299 items_per_second=8.98286k/s
// MatrixMul3/512/256/1       431551 ns       431487 ns         1623 items_per_second=2.31757k/s
// MatrixMul3/512/512/1       856521 ns       856435 ns          814 items_per_second=1.16763k/s
// MatrixMul3/512/1024/1     1721905 ns      1721668 ns          409 items_per_second=580.832/s
// MatrixMul3/512/2048/1     3427994 ns      3427439 ns          204 items_per_second=291.763/s
// MatrixMul3/64/256/32      1723179 ns      1722987 ns          406 items_per_second=580.388/s
// MatrixMul3/64/256/64      3450749 ns      3450406 ns          203 items_per_second=289.821/s
// MatrixMul3/64/256/128     6896719 ns      6895785 ns          101 items_per_second=145.016/s
// MatrixMul3/64/256/512    27919240 ns     27914426 ns           25 items_per_second=35.8238/s
// MatrixMul3/128/256/32     3457250 ns      3456687 ns          203 items_per_second=289.294/s
// MatrixMul3/128/256/64     6903948 ns      6902900 ns          100 items_per_second=144.867/s
// MatrixMul3/128/256/128   13807285 ns     13804911 ns           50 items_per_second=72.438/s
// MatrixMul3/128/256/512   55710867 ns     55701494 ns           12 items_per_second=17.9528/s
// MatrixMul3/256/256/32     6906703 ns      6905944 ns          100 items_per_second=144.803/s
// MatrixMul3/256/256/64    13793217 ns     13791186 ns           51 items_per_second=72.5101/s
// MatrixMul3/256/256/128   27580987 ns     27578309 ns           25 items_per_second=36.2604/s
// MatrixMul3/256/256/512  111919732 ns    111895506 ns            6 items_per_second=8.93691/s
// MatrixMul3/512/256/32    13789928 ns     13787942 ns           51 items_per_second=72.5271/s
// MatrixMul3/512/256/64    27926471 ns     27924927 ns           25 items_per_second=35.8103/s
// MatrixMul3/512/256/128   55343544 ns     55328991 ns           12 items_per_second=18.0737/s
// MatrixMul3/512/256/512  222171427 ns    222147059 ns            3 items_per_second=4.50152/s
// MatrixMul3/64/512/32      3431270 ns      3430779 ns          204 items_per_second=291.479/s
// MatrixMul3/64/512/64      6862159 ns      6861322 ns          102 items_per_second=145.745/s
// MatrixMul3/64/512/128    13723187 ns     13721447 ns           51 items_per_second=72.8786/s
// MatrixMul3/64/512/512    55796384 ns     55787264 ns           12 items_per_second=17.9252/s
// MatrixMul3/128/512/32     6863031 ns      6861669 ns          102 items_per_second=145.737/s
// MatrixMul3/128/512/64    13722977 ns     13721048 ns           51 items_per_second=72.8807/s
// MatrixMul3/128/512/128   27451040 ns     27448295 ns           25 items_per_second=36.4321/s
// MatrixMul3/128/512/512  111597169 ns    111581313 ns            6 items_per_second=8.96207/s
// MatrixMul3/256/512/32    13725884 ns     13724241 ns           51 items_per_second=72.8638/s
// MatrixMul3/256/512/64    27532854 ns     27527905 ns           26 items_per_second=36.3268/s
// MatrixMul3/256/512/128   54904911 ns     54897602 ns           13 items_per_second=18.2157/s
// MatrixMul3/256/512/512  222959788 ns    222936200 ns            3 items_per_second=4.48559/s
// MatrixMul3/512/512/32    27491247 ns     27484819 ns           25 items_per_second=36.3837/s
// MatrixMul3/512/512/64    55062282 ns     55043619 ns           13 items_per_second=18.1674/s
// MatrixMul3/512/512/128  109848638 ns    109837318 ns            6 items_per_second=9.10437/s
// MatrixMul3/512/512/512  446769163 ns    446711860 ns            2 items_per_second=2.23858/s