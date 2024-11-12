#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  char* a = (char*)malloc(sizeof(char) * array_size);
  char* b = (char*)malloc(sizeof(char) * array_size);
  char* c = (char*)malloc(sizeof(char) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
}

static void ShortPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  short* a = (short*)malloc(sizeof(short) * array_size);
  short* b = (short*)malloc(sizeof(short) * array_size);
  short* c = (short*)malloc(sizeof(short) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
}

static void IntPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  int* a = (int*)malloc(sizeof(int) * array_size);
  int* b = (int*)malloc(sizeof(int) * array_size);
  int* c = (int*)malloc(sizeof(int) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q]; 
    }
    escape(&c);
  }
}

static void LongLongPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  long long* a = (long long*)malloc(sizeof(long long) * array_size);
  long long* b = (long long*)malloc(sizeof(long long) * array_size);
  long long* c = (long long*)malloc(sizeof(long long) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
}

static void FloatPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  float* a = (float*)malloc(sizeof(float) * array_size);
  float* b = (float*)malloc(sizeof(float) * array_size);
  float* c = (float*)malloc(sizeof(float) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
}

static void DoublePLus(benchmark::State& state) {
  const int array_size = state.range(0);
  double* a = (double*)malloc(sizeof(double) * array_size);
  double* b = (double*)malloc(sizeof(double) * array_size);
  double* c = (double*)malloc(sizeof(double) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
}

static void LongDoublePLus(benchmark::State& state) {
  const int array_size = state.range(0);
  long double* a = (long double*)malloc(sizeof(long double) * array_size);
  long double* b = (long double*)malloc(sizeof(long double) * array_size);
  long double* c = (long double*)malloc(sizeof(long double) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
}

BENCHMARK(CharPLus)->RangeMultiplier(4)->Range(1, 1<<16);
// ---------------------------------------------------------
// Benchmark               Time             CPU   Iterations
// ---------------------------------------------------------
// CharPLus/1          0.359 ns        0.359 ns   1944678108
// CharPLus/4           1.46 ns         1.46 ns    475769685
// CharPLus/16          7.27 ns         7.28 ns     96083077
// CharPLus/64          30.4 ns         30.5 ns     22900304
// CharPLus/256          127 ns          127 ns      5494027
// CharPLus/1024         498 ns          498 ns      1393707
// CharPLus/4096        1987 ns         1988 ns       352160
// CharPLus/16384       8049 ns         8056 ns        86779
// CharPLus/65536      32484 ns        32516 ns        21617
// only loop unroll that is applied

BENCHMARK(ShortPLus)->RangeMultiplier(4)->Range(1, 1<<16);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// ShortPLus/1          0.060 ns        0.060 ns   11669133521
// ShortPLus/4           1.73 ns         1.73 ns    404254494
// ShortPLus/16          2.92 ns         2.92 ns    238376320
// ShortPLus/64          4.99 ns         4.99 ns    139282786
// ShortPLus/256         12.3 ns         12.3 ns     56635078
// ShortPLus/1024        47.5 ns         47.5 ns     14724241
// ShortPLus/4096         186 ns          186 ns      3755943
// ShortPLus/16384       1012 ns         1013 ns       690481
// ShortPLus/65536       4334 ns         4337 ns       161460
// the vectorlize and loop unrolls in applied.

BENCHMARK(IntPLus)->RangeMultiplier(4)->Range(1, 1<<16);
// --------------------------------------------------------
// Benchmark              Time             CPU   Iterations
// --------------------------------------------------------
// IntPLus/1          0.060 ns        0.060 ns   11708413601
// IntPLus/4           1.76 ns         1.76 ns    404891539
// IntPLus/16          3.06 ns         3.06 ns    228541683
// IntPLus/64          6.33 ns         6.32 ns    109866222
// IntPLus/256         23.6 ns         23.5 ns     29725289
// IntPLus/1024        93.7 ns         93.6 ns      7457593
// IntPLus/4096         378 ns          377 ns      1853543
// IntPLus/16384       2046 ns         2043 ns       338961
// IntPLus/65536       8767 ns         8756 ns        777338
// the vectorlize and loop unrolls in applied.

BENCHMARK(LongLongPLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------
// Benchmark                  Time             CPU   Iterations
// ------------------------------------------------------------
// LongLongPLus/1         0.060 ns        0.060 ns   11574675965
// LongLongPLus/4          1.62 ns         1.62 ns    381314712
// LongLongPLus/16         3.66 ns         3.66 ns    191827288
// LongLongPLus/64         11.4 ns         11.4 ns     61548886
// LongLongPLus/256        46.6 ns         46.6 ns     15000253
// LongLongPLus/1024        186 ns          186 ns      3766327
// the vectorlize and loop unrolls in applied.

BENCHMARK(FloatPLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ---------------------------------------------------------
// Benchmark               Time             CPU   Iterations
// ---------------------------------------------------------
// FloatPLus/1         0.060 ns        0.060 ns   11593399666
// FloatPLus/4          1.70 ns         1.70 ns    412511052
// FloatPLus/16         3.03 ns         3.03 ns    231109266
// FloatPLus/64         6.32 ns         6.32 ns    109826035
// FloatPLus/256        23.4 ns         23.4 ns     29048556
// FloatPLus/1024       93.2 ns         93.2 ns      7503030
// the vectorlize and loop unrolls in applied.

BENCHMARK(DoublePLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// DoublePLus/1         0.060 ns        0.060 ns   11534330600
// DoublePLus/4          1.61 ns         1.61 ns    432587720
// DoublePLus/16         3.64 ns         3.64 ns    191790270
// DoublePLus/64         11.3 ns         11.3 ns     61747149
// DoublePLus/256        46.5 ns         46.5 ns     14857035
// DoublePLus/1024        186 ns          186 ns      3768157
// the vectorlize and loop unrolls in applied.

BENCHMARK(LongDoublePLus)->RangeMultiplier(4)->Range(1, 1<<10);
// --------------------------------------------------------------
// Benchmark                    Time             CPU   Iterations
// --------------------------------------------------------------
// LongDoublePLus/1         0.060 ns        0.060 ns   11412105391
// LongDoublePLus/4          10.1 ns         10.1 ns     69061660
// LongDoublePLus/16         48.3 ns         48.3 ns     14534162
// LongDoublePLus/64          199 ns          199 ns      3505418
// LongDoublePLus/256         803 ns          804 ns       865540
// LongDoublePLus/1024       3227 ns         3228 ns       216774
// only loop unroll that is applied

// the vectorize and loop unrolls need -funroll-loops -ftree-vectorize flag to force it to happen.

BENCHMARK_MAIN();