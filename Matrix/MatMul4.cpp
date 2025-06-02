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

template<int d1,int d2,int d3>
static void MatrixMul11(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d1,d2>();
  const int _d2 = GetColSizeFloat(d2);
  Tensor3 b = Create0<_d2,d3>();
  Tensor3 c = Create0<d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < d2 * d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,a);
  FromArray<_d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul12(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d1,d2>();
  const int _d2 = GetColSizeFloat(d2);
  Tensor3 b = Create0<_d2,d3>();
  Tensor3 c = Create0<d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < d2;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,a);
  FromArray<_d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus2<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul14(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d1,d2>();
  const int _d2 = GetColSizeFloat(d2);
  Tensor3 b = Create0<_d2,d3>();
  Tensor3 c = Create0<d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < _d2 * d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,a);
  FromArray<_d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus4<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul15(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d2,d1>();
  Tensor3 b = Create0<d2,d3>();
  const int _d1 = GetColSizeFloat(d1);
  Tensor3 c = Create0<_d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < d2 * d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d2,d1>(in1,a);
  FromArray<d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus5<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul17(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d2,d1>();
  Tensor3 b = Create0<d2,d3>();
  const int _d1 = GetColSizeFloat(d1);
  Tensor3 c = Create0<_d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < d2 * d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d2,d1>(in1,a);
  FromArray<d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus7<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul18(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d1,d2>();
  const int _d2 = GetColSizeFloat(d2);
  Tensor3 b = Create0<_d2,d3>();
  Tensor3 c = Create0<d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < _d2 * d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,a);
  FromArray<_d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus8<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul19(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  Tensor3 a = Create0<d1,d2>();
  const int _d2 = GetColSizeFloat(d2);
  Tensor3 b = Create0<_d2,d3>();
  Tensor3 c = Create0<d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < _d2 * d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,a);
  FromArray<_d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus8<d1,d2,d3>(a,b,c);
    escape(c);
  }
  state.SetItemsProcessed(state.iterations());
}

template<int d1,int d2,int d3>
static void MatrixMul20(benchmark::State& state) {
  // L1_CACHE 256 * 1024
  // L1_CACHE LINE 64
  float* in1 = (float*)malloc(sizeof(float) * d2 * d1);
  float* in2 = (float*)malloc(sizeof(float) * d2 * d3);
  const int _d3 = GetColSizeFloat(d3);
  Tensor3 a = Create0<d1,d2>();
  Tensor3 b = Create0<d2,d3>();
  Tensor3 c = Create0<d1,d3>();
  for(int w = 0;w < d2 * d1;w++) {
    in1[w] = randomFloat(-1,1);
  }
  for(int q = 0;q < d2 * _d3;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,a);
  FromArray<d2,d3>(in2,b);
  for(auto _ : state) {
    MatMulPlus10<d1,d2,d3>(a,b,c);
    escape(c);
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

#define BENCHMARK_TEMP_SMMALL(func) \
BENCHMARK_TEMPLATE(func,1024,1024,128);\
BENCHMARK_TEMPLATE(func,1024,1024,512);\
BENCHMARK_TEMPLATE(func,1024,1024,1024);

#define BENCHMARK_TEMP(func) \
BENCHMARK_TEMPLATE(func,8,8,8);\
BENCHMARK_TEMPLATE(func,8,8,32);\
BENCHMARK_TEMPLATE(func,8,8,128);\
BENCHMARK_TEMPLATE(func,8,8,512);\
BENCHMARK_TEMPLATE(func,8,8,1024);\
BENCHMARK_TEMPLATE(func,8,32,8);\
BENCHMARK_TEMPLATE(func,8,32,32);\
BENCHMARK_TEMPLATE(func,8,32,128);\
BENCHMARK_TEMPLATE(func,8,32,512);\
BENCHMARK_TEMPLATE(func,8,32,1024);\
BENCHMARK_TEMPLATE(func,8,128,8);\
BENCHMARK_TEMPLATE(func,8,128,32);\
BENCHMARK_TEMPLATE(func,8,128,128);\
BENCHMARK_TEMPLATE(func,8,128,512);\
BENCHMARK_TEMPLATE(func,8,128,1024);\
BENCHMARK_TEMPLATE(func,8,512,8);\
BENCHMARK_TEMPLATE(func,8,512,32);\
BENCHMARK_TEMPLATE(func,8,512,128);\
BENCHMARK_TEMPLATE(func,8,512,512);\
BENCHMARK_TEMPLATE(func,8,512,1024);\
BENCHMARK_TEMPLATE(func,8,1024,8);\
BENCHMARK_TEMPLATE(func,8,1024,32);\
BENCHMARK_TEMPLATE(func,8,1024,128);\
BENCHMARK_TEMPLATE(func,8,1024,512);\
BENCHMARK_TEMPLATE(func,8,1024,1024);\
BENCHMARK_TEMPLATE(func,32,8,8);\
BENCHMARK_TEMPLATE(func,32,8,32);\
BENCHMARK_TEMPLATE(func,32,8,128);\
BENCHMARK_TEMPLATE(func,32,8,512);\
BENCHMARK_TEMPLATE(func,32,8,1024);\
BENCHMARK_TEMPLATE(func,32,32,8);\
BENCHMARK_TEMPLATE(func,32,32,32);\
BENCHMARK_TEMPLATE(func,32,32,128);\
BENCHMARK_TEMPLATE(func,32,32,512);\
BENCHMARK_TEMPLATE(func,32,32,1024);\
BENCHMARK_TEMPLATE(func,32,128,8);\
BENCHMARK_TEMPLATE(func,32,128,32);\
BENCHMARK_TEMPLATE(func,32,128,128);\
BENCHMARK_TEMPLATE(func,32,128,512);\
BENCHMARK_TEMPLATE(func,32,128,1024);\
BENCHMARK_TEMPLATE(func,32,512,8);\
BENCHMARK_TEMPLATE(func,32,512,32);\
BENCHMARK_TEMPLATE(func,32,512,128);\
BENCHMARK_TEMPLATE(func,32,512,512);\
BENCHMARK_TEMPLATE(func,32,512,1024);\
BENCHMARK_TEMPLATE(func,32,1024,8);\
BENCHMARK_TEMPLATE(func,32,1024,32);\
BENCHMARK_TEMPLATE(func,32,1024,128);\
BENCHMARK_TEMPLATE(func,32,1024,512);\
BENCHMARK_TEMPLATE(func,32,1024,1024);\
BENCHMARK_TEMPLATE(func,128,8,8);\
BENCHMARK_TEMPLATE(func,128,8,32);\
BENCHMARK_TEMPLATE(func,128,8,128);\
BENCHMARK_TEMPLATE(func,128,8,512);\
BENCHMARK_TEMPLATE(func,128,8,1024);\
BENCHMARK_TEMPLATE(func,128,32,8);\
BENCHMARK_TEMPLATE(func,128,32,32);\
BENCHMARK_TEMPLATE(func,128,32,128);\
BENCHMARK_TEMPLATE(func,128,32,512);\
BENCHMARK_TEMPLATE(func,128,32,1024);\
BENCHMARK_TEMPLATE(func,128,128,8);\
BENCHMARK_TEMPLATE(func,128,128,32);\
BENCHMARK_TEMPLATE(func,128,128,128);\
BENCHMARK_TEMPLATE(func,128,128,512);\
BENCHMARK_TEMPLATE(func,128,128,1024);\
BENCHMARK_TEMPLATE(func,128,512,8);\
BENCHMARK_TEMPLATE(func,128,512,32);\
BENCHMARK_TEMPLATE(func,128,512,128);\
BENCHMARK_TEMPLATE(func,128,512,512);\
BENCHMARK_TEMPLATE(func,128,512,1024);\
BENCHMARK_TEMPLATE(func,128,1024,8);\
BENCHMARK_TEMPLATE(func,128,1024,32);\
BENCHMARK_TEMPLATE(func,128,1024,128);\
BENCHMARK_TEMPLATE(func,128,1024,512);\
BENCHMARK_TEMPLATE(func,128,1024,1024);\
BENCHMARK_TEMPLATE(func,512,8,8);\
BENCHMARK_TEMPLATE(func,512,8,32);\
BENCHMARK_TEMPLATE(func,512,8,128);\
BENCHMARK_TEMPLATE(func,512,8,512);\
BENCHMARK_TEMPLATE(func,512,8,1024);\
BENCHMARK_TEMPLATE(func,512,32,8);\
BENCHMARK_TEMPLATE(func,512,32,32);\
BENCHMARK_TEMPLATE(func,512,32,128);\
BENCHMARK_TEMPLATE(func,512,32,512);\
BENCHMARK_TEMPLATE(func,512,32,1024);\
BENCHMARK_TEMPLATE(func,512,128,8);\
BENCHMARK_TEMPLATE(func,512,128,32);\
BENCHMARK_TEMPLATE(func,512,128,128);\
BENCHMARK_TEMPLATE(func,512,128,512);\
BENCHMARK_TEMPLATE(func,512,128,1024);\
BENCHMARK_TEMPLATE(func,512,512,8);\
BENCHMARK_TEMPLATE(func,512,512,32);\
BENCHMARK_TEMPLATE(func,512,512,128);\
BENCHMARK_TEMPLATE(func,512,512,512);\
BENCHMARK_TEMPLATE(func,512,512,1024);\
BENCHMARK_TEMPLATE(func,512,1024,8);\
BENCHMARK_TEMPLATE(func,512,1024,32);\
BENCHMARK_TEMPLATE(func,512,1024,128);\
BENCHMARK_TEMPLATE(func,512,1024,512);\
BENCHMARK_TEMPLATE(func,512,1024,1024);\
BENCHMARK_TEMPLATE(func,1024,8,8);\
BENCHMARK_TEMPLATE(func,1024,8,32);\
BENCHMARK_TEMPLATE(func,1024,8,128);\
BENCHMARK_TEMPLATE(func,1024,8,512);\
BENCHMARK_TEMPLATE(func,1024,8,1024);\
BENCHMARK_TEMPLATE(func,1024,32,8);\
BENCHMARK_TEMPLATE(func,1024,32,32);\
BENCHMARK_TEMPLATE(func,1024,32,128);\
BENCHMARK_TEMPLATE(func,1024,32,512);\
BENCHMARK_TEMPLATE(func,1024,32,1024);\
BENCHMARK_TEMPLATE(func,1024,128,8);\
BENCHMARK_TEMPLATE(func,1024,128,32);\
BENCHMARK_TEMPLATE(func,1024,128,128);\
BENCHMARK_TEMPLATE(func,1024,128,512);\
BENCHMARK_TEMPLATE(func,1024,128,1024);\
BENCHMARK_TEMPLATE(func,1024,512,8);\
BENCHMARK_TEMPLATE(func,1024,512,32);\
BENCHMARK_TEMPLATE(func,1024,512,128);\
BENCHMARK_TEMPLATE(func,1024,512,512);\
BENCHMARK_TEMPLATE(func,1024,512,1024);\
BENCHMARK_TEMPLATE(func,1024,1024,8);\
BENCHMARK_TEMPLATE(func,1024,1024,32);\
BENCHMARK_TEMPLATE(func,1024,1024,128);\
BENCHMARK_TEMPLATE(func,1024,1024,512);\
BENCHMARK_TEMPLATE(func,1024,1024,1024);\

//BENCHMARK(MatrixMul1)->Apply(CustomArgs);
//BENCHMARK(MatrixMul3)->Apply(CustomArgs);
//BENCHMARK(MatrixMul2)->Apply(CustomArgs);
//BENCHMARK(MatrixMul4)->Apply(CustomArgs);
//BENCHMARK(MatrixMul5)->Apply(CustomArgs);
//BENCHMARK(MatrixMul6)->Apply(CustomArgs);
// BENCHMARK(MatrixMul7)->Apply(CustomArgs);
// BENCHMARK_TEMP_SMMALL(MatrixMul11);
// BENCHMARK_TEMP_SMMALL(MatrixMul12);
// BENCHMARK_TEMP_SMMALL(MatrixMul14);
// BENCHMARK_TEMP_SMMALL(MatrixMul15);
// BENCHMARK_TEMP(MatrixMul17);
//BENCHMARK_TEMP(MatrixMul18);
// BENCHMARK_TEMP(MatrixMul19);
BENCHMARK_TEMP(MatrixMul20);


BENCHMARK_MAIN();