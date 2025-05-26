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

static void PlusMatrix1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  Matrix m(d1, d2);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d1;q++) {
    for(int w = 0;w < d2;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    out = t + m;
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
static void PlusFloat1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  const float f = randomFloat(-1,1);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    out = t + f;
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void PlusMatrix2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float* in2 = (float*)malloc(sizeof(float) * d1 * d2);
  Tensor3 t = Create<d1,d2>();
  Tensor3 m = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  for(int q = 0;q < d1 * d2;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  FromArray<d1,d2>(in2,m);
  for(auto _ : state) {
    Plus<d1,d2>(t,m,out);
    escape(out);
  }
  std::free(in1);
  std::free(in2);
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void PlusFloat2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float f = randomFloat(-1,1);
  Tensor3 t = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  for(auto _ : state) {
    Plus<d1,d2>(t,f,out);
    escape(out);
  }
  std::free(in1);
  state.SetItemsProcessed(state.iterations());
}

static void SubMatrix1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  Matrix m(d1, d2);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d1;q++) {
    for(int w = 0;w < d2;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    out = t - m;
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
static void SubFloat1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  const float f = randomFloat(-1,1);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    out = t - f;
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void SubMatrix2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float* in2 = (float*)malloc(sizeof(float) * d1 * d2);
  Tensor3 t = Create<d1,d2>();
  Tensor3 m = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  for(int q = 0;q < d1 * d2;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  FromArray<d1,d2>(in2,m);
  for(auto _ : state) {
    Sub<d1,d2>(t,m,out);
    escape(out);
  }
  std::free(in1);
  std::free(in2);
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void SubFloat2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float f = randomFloat(-1,1);
  Tensor3 t = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  for(auto _ : state) {
    Sub<d1,d2>(t,f,out);
    escape(out);
  }
  std::free(in1);
  state.SetItemsProcessed(state.iterations());
}

static void MulMatrix1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  Matrix m(d1, d2);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d1;q++) {
    for(int w = 0;w < d2;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d1;q++) {
      for(int w = 0;w < d2;w++) {
        out[q][w] = t[q][w] * m[q][w];
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
static void MulFloat1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  const float f = randomFloat(-1,1);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d1;q++) {
      for(int w = 0;w < d2;w++) {
        out[q][w] = t[q][w] * f;
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void MulMatrix2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float* in2 = (float*)malloc(sizeof(float) * d1 * d2);
  Tensor3 t = Create<d1,d2>();
  Tensor3 m = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  for(int q = 0;q < d1 * d2;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  FromArray<d1,d2>(in2,m);
  for(auto _ : state) {
    Mul<d1,d2>(t,m,out);
    escape(out);
  }
  std::free(in1);
  std::free(in2);
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void MulFloat2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float f = randomFloat(-1,1);
  Tensor3 t = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  for(auto _ : state) {
    Mul<d1,d2>(t,f,out);
    escape(out);
  }
  std::free(in1);
  state.SetItemsProcessed(state.iterations());
}

static void DivMatrix1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  Matrix m(d1, d2);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(int q = 0;q < d1;q++) {
    for(int w = 0;w < d2;w++) {
      m[q][w] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d1;q++) {
      for(int w = 0;w < d2;w++) {
        out[q][w] = t[q][w] / m[q][w];
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
static void DivFloat1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Matrix t(d1, d2);
  const float f = randomFloat(-1,1);
  Matrix out(d1, d2);
  for(int w = 0;w < d1;w++) {
    for(int e = 0;e < d2;e++) {
      t[w][e] = randomFloat(-1,1);
    }
  }
  for(auto _ : state) {
    for(int q = 0;q < d1;q++) {
      for(int w = 0;w < d2;w++) {
        out[q][w] = t[q][w] / f;
      }
    }
    escape(out.data.data());
  }
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void DivMatrix2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float* in2 = (float*)malloc(sizeof(float) * d1 * d2);
  Tensor3 t = Create<d1,d2>();
  Tensor3 m = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  for(int q = 0;q < d1 * d2;q++) {
    in2[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  FromArray<d1,d2>(in2,m);
  for(auto _ : state) {
    Div<d1,d2>(t,m,out);
    escape(out);
  }
  std::free(in1);
  std::free(in2);
  state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2>
static void DivFloat2(benchmark::State& state) {
  float* in1 = (float*)malloc(sizeof(float) * d1 * d2);
  float f = randomFloat(-1,1);
  Tensor3 t = Create<d1,d2>();
  Tensor3 out = Create<d1,d2>();
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = randomFloat(-1,1);
  }
  FromArray<d1,d2>(in1,t);
  for(auto _ : state) {
    Div<d1,d2>(t,f,out);
    escape(out);
  }
  std::free(in1);
  state.SetItemsProcessed(state.iterations());
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  b->Args({1, 2048});
  b->Args({64, 2048});
  b->Args({2048, 64});
  b->Args({2048, 512});
}

#define BENCHMARK_TEMP(func) BENCHMARK_TEMPLATE(func, 1, 2048); BENCHMARK_TEMPLATE(func, 64, 2048); BENCHMARK_TEMPLATE(func, 2048, 64); BENCHMARK_TEMPLATE(func, 2048, 512);

BENCHMARK(PlusMatrix1)->Apply(CustomArgs);
BENCHMARK(PlusFloat1)->Apply(CustomArgs);
BENCHMARK_TEMP(PlusMatrix2);
BENCHMARK_TEMP(PlusMatrix2);

BENCHMARK(SubMatrix1)->Apply(CustomArgs);
BENCHMARK(SubFloat1)->Apply(CustomArgs);
BENCHMARK_TEMP(SubMatrix2);
BENCHMARK_TEMP(SubFloat2);

BENCHMARK(MulMatrix1)->Apply(CustomArgs);
BENCHMARK(MulFloat1)->Apply(CustomArgs);
BENCHMARK_TEMP(MulMatrix2);
BENCHMARK_TEMP(MulFloat2);

BENCHMARK(DivMatrix1)->Apply(CustomArgs);
BENCHMARK(DivFloat1)->Apply(CustomArgs);
BENCHMARK_TEMP(DivMatrix2);
BENCHMARK_TEMP(DivFloat2);

BENCHMARK_MAIN();

// -------------------------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------
// PlusMatrix1/1/2048          276 ns          276 ns      2525230 items_per_second=3.624M/s
// PlusMatrix1/64/2048       41427 ns        41422 ns        17139 items_per_second=24.1416k/s
// PlusMatrix1/2048/64       41281 ns        41279 ns        17582 items_per_second=24.2251k/s
// PlusMatrix1/2048/512    2627621 ns      2627162 ns          260 items_per_second=380.639/s
// PlusFloat1/1/2048           335 ns          335 ns      2084538 items_per_second=2.98692M/s
// PlusFloat1/64/2048        33699 ns        33695 ns        20880 items_per_second=29.678k/s
// PlusFloat1/2048/64        34034 ns        34028 ns        20785 items_per_second=29.3875k/s
// PlusFloat1/2048/512     1893241 ns      1892615 ns          364 items_per_second=528.369/s
// PlusMatrix2/1/2048         96.8 ns         96.8 ns      7201855 items_per_second=10.3314M/s
// PlusMatrix2/64/2048       18141 ns        18135 ns        38358 items_per_second=55.1421k/s
// PlusMatrix2/2048/64       18540 ns        18537 ns        38038 items_per_second=53.9466k/s
// PlusMatrix2/2048/512    1283405 ns      1282991 ns          532 items_per_second=779.429/s
// PlusFloat2/1/2048          65.8 ns         65.8 ns     10557111 items_per_second=15.2042M/s
// PlusFloat2/64/2048        14762 ns        14760 ns        47686 items_per_second=67.749k/s
// PlusFloat2/2048/64        14295 ns        14292 ns        48528 items_per_second=69.9673k/s
// PlusFloat2/2048/512     1047879 ns      1047440 ns          683 items_per_second=954.709/s
// SubMatrix1/1/2048           266 ns          265 ns      2628477 items_per_second=3.76706M/s
// SubMatrix1/64/2048        40416 ns        40412 ns        17638 items_per_second=24.7449k/s
// SubMatrix1/2048/64        39396 ns        39382 ns        17767 items_per_second=25.3925k/s
// SubMatrix1/2048/512     2448601 ns      2448023 ns          290 items_per_second=408.493/s
// SubFloat1/1/2048            323 ns          323 ns      2167111 items_per_second=3.09474M/s
// SubFloat1/64/2048         33116 ns        33112 ns        21129 items_per_second=30.2004k/s
// SubFloat1/2048/64         32137 ns        32129 ns        21434 items_per_second=31.1243k/s
// SubFloat1/2048/512      1885494 ns      1884791 ns          371 items_per_second=530.563/s
// SubMatrix2/1/2048          96.3 ns         96.3 ns      7247445 items_per_second=10.3821M/s
// SubMatrix2/64/2048        19859 ns        19853 ns        36335 items_per_second=50.3691k/s
// SubMatrix2/2048/64        19762 ns        19757 ns        36027 items_per_second=50.6141k/s
// SubMatrix2/2048/512     1413558 ns      1413059 ns          522 items_per_second=707.684/s
// SubFloat2/1/2048           64.8 ns         64.8 ns     10741811 items_per_second=15.4317M/s
// SubFloat2/64/2048         13631 ns        13631 ns        52644 items_per_second=73.3641k/s
// SubFloat2/2048/64         13238 ns        13237 ns        49723 items_per_second=75.5476k/s
// SubFloat2/2048/512      1042421 ns      1041933 ns          646 items_per_second=959.755/s
// MulMatrix1/1/2048          8245 ns         8244 ns        84287 items_per_second=121.299k/s
// MulMatrix1/64/2048       530154 ns       530058 ns         1315 items_per_second=1.88658k/s
// MulMatrix1/2048/64       541409 ns       541310 ns         1285 items_per_second=1.84737k/s
// MulMatrix1/2048/512     4270634 ns      4269370 ns          164 items_per_second=234.227/s
// MulFloat1/1/2048           5383 ns         5383 ns       129561 items_per_second=185.779k/s
// MulFloat1/64/2048        346237 ns       346180 ns         2022 items_per_second=2.88867k/s
// MulFloat1/2048/64        357644 ns       357403 ns         1956 items_per_second=2.79796k/s
// MulFloat1/2048/512      2782273 ns      2781464 ns          251 items_per_second=359.523/s
// MulMatrix2/1/2048          96.5 ns         96.5 ns      7262640 items_per_second=10.3667M/s
// MulMatrix2/64/2048        18349 ns        18345 ns        37993 items_per_second=54.5113k/s
// MulMatrix2/2048/64        18574 ns        18571 ns        38045 items_per_second=53.8479k/s
// MulMatrix2/2048/512     1294587 ns      1294180 ns          528 items_per_second=772.69/s
// MulFloat2/1/2048           65.6 ns         65.6 ns     10662165 items_per_second=15.2479M/s
// MulFloat2/64/2048         14064 ns        14062 ns        49286 items_per_second=71.1127k/s
// MulFloat2/2048/64         14154 ns        14152 ns        49227 items_per_second=70.6621k/s
// MulFloat2/2048/512      1004836 ns      1004411 ns          687 items_per_second=995.609/s
// DivMatrix1/1/2048          8251 ns         8249 ns        84779 items_per_second=121.231k/s
// DivMatrix1/64/2048       529518 ns       529351 ns         1320 items_per_second=1.88911k/s
// DivMatrix1/2048/64       539249 ns       539154 ns         1285 items_per_second=1.85476k/s
// DivMatrix1/2048/512     4254140 ns      4252799 ns          165 items_per_second=235.139/s
// DivFloat1/1/2048           5497 ns         5496 ns       126785 items_per_second=181.938k/s
// DivFloat1/64/2048        352925 ns       352854 ns         1988 items_per_second=2.83403k/s
// DivFloat1/2048/64        365210 ns       365155 ns         1915 items_per_second=2.73856k/s
// DivFloat1/2048/512      2845803 ns      2845072 ns          245 items_per_second=351.485/s
// DivMatrix2/1/2048           217 ns          217 ns      3223900 items_per_second=4.60839M/s
// DivMatrix2/64/2048        22940 ns        22936 ns        29522 items_per_second=43.599k/s
// DivMatrix2/2048/64        23409 ns        23408 ns        29931 items_per_second=42.7213k/s
// DivMatrix2/2048/512     1323115 ns      1322757 ns          502 items_per_second=755.997/s
// DivFloat2/1/2048            217 ns          217 ns      3228646 items_per_second=4.61321M/s
// DivFloat2/64/2048         15704 ns        15702 ns        45197 items_per_second=63.6857k/s
// DivFloat2/2048/64         15370 ns        15366 ns        45245 items_per_second=65.0798k/s
// DivFloat2/2048/512      1005300 ns      1004988 ns          693 items_per_second=995.037/s