#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  char* a = (char*)malloc(sizeof(char) * array_size);
  char* b = (char*)malloc(sizeof(char) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      char c = a[q] + b[q];
      escape(&c);
    }
  }
}

static void ShortPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  short* a = (short*)malloc(sizeof(short) * array_size);
  short* b = (short*)malloc(sizeof(short) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      short c = a[q] + b[q];
      escape(&c);
    }
  }
}

static void IntPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  int* a = (int*)malloc(sizeof(int) * array_size);
  int* b = (int*)malloc(sizeof(int) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      int c = a[q] + b[q];
      escape(&c);
    }
  }
}

static void LongLongPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  long long* a = (long long*)malloc(sizeof(long long) * array_size);
  long long* b = (long long*)malloc(sizeof(long long) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      long long c = a[q] + b[q];
      escape(&c);
    }
  }
}

static void FloatPLus(benchmark::State& state) {
  const int array_size = state.range(0);
  float* a = (float*)malloc(sizeof(float) * array_size);
  float* b = (float*)malloc(sizeof(float) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      float c = a[q] + b[q];
      escape(&c);
    }
  }
}

static void DoublePLus(benchmark::State& state) {
  const int array_size = state.range(0);
  double* a = (double*)malloc(sizeof(double) * array_size);
  double* b = (double*)malloc(sizeof(double) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      double c = a[q] + b[q];
      escape(&c);
    }
  }
}

static void LongDoublePLus(benchmark::State& state) {
  const int array_size = state.range(0);
  long double* a = (long double*)malloc(sizeof(long double) * array_size);
  long double* b = (long double*)malloc(sizeof(long double) * array_size);
  for(auto _ : state) {
    for(int q = 0;q < array_size - 1;q++) {
      long double c = a[q] + b[q];
      escape(&c);
    }
  }
}

BENCHMARK(CharPLus)->RangeMultiplier(4)->Range(1, 1<<16);
// ---------------------------------------------------------
// Benchmark               Time             CPU   Iterations
// ---------------------------------------------------------
// CharPLus/1          0.060 ns        0.060 ns   11617221834
// CharPLus/4           1.73 ns         1.73 ns    405133971
// CharPLus/16          5.09 ns         5.09 ns    137034690
// CharPLus/64          17.8 ns         17.8 ns     39319573
// CharPLus/256         67.8 ns         67.8 ns     10326954
// CharPLus/1024         269 ns          269 ns      2567417
// CharPLus/4096        1061 ns         1061 ns       659860
// CharPLus/16384       4400 ns         4402 ns       158627
// CharPLus/65536      17780 ns        17788 ns        39281
// loop unroll and no vectorize

BENCHMARK(ShortPLus)->RangeMultiplier(4)->Range(1, 1<<16);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// ShortPLus/1          0.060 ns        0.060 ns   11571483883
// ShortPLus/4           1.71 ns         1.71 ns    410394043
// ShortPLus/16          5.08 ns         5.08 ns    136597749
// ShortPLus/64          17.8 ns         17.8 ns     39202993
// ShortPLus/256         67.1 ns         67.2 ns     10354825
// ShortPLus/1024         271 ns          271 ns      2584551
// ShortPLus/4096        1095 ns         1096 ns       634744
// ShortPLus/16384       4496 ns         4498 ns       155527
// ShortPLus/65536      18989 ns        18995 ns        36837
// loop unroll and no vectorize

BENCHMARK(IntPLus)->RangeMultiplier(4)->Range(1, 1<<16);
// --------------------------------------------------------
// Benchmark              Time             CPU   Iterations
// --------------------------------------------------------
// IntPLus/1          0.060 ns        0.060 ns   11381645045
// IntPLus/4           1.71 ns         1.71 ns    411384645
// IntPLus/16          5.12 ns         5.12 ns    134855329
// IntPLus/64          17.9 ns         17.9 ns     38992168
// IntPLus/256         67.5 ns         67.6 ns     10297501
// IntPLus/1024         271 ns          271 ns      2550013
// IntPLus/4096        1207 ns         1207 ns       580011
// IntPLus/16384       4942 ns         4944 ns       141697
// IntPLus/65536      19607 ns        19614 ns        35658
// loop unroll and no vectorize

BENCHMARK(LongLongPLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------
// Benchmark                  Time             CPU   Iterations
// ------------------------------------------------------------
// LongLongPLus/1         0.061 ns        0.061 ns   11336935672
// LongLongPLus/4          1.73 ns         1.73 ns    401513292
// LongLongPLus/16         5.17 ns         5.17 ns    133734905
// LongLongPLus/64         18.1 ns         18.1 ns     38899826
// LongLongPLus/256        68.0 ns         68.0 ns     10255787
// LongLongPLus/1024        298 ns          299 ns      2340935
// loop unroll and no vectorize

BENCHMARK(FloatPLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ---------------------------------------------------------
// Benchmark               Time             CPU   Iterations
// ---------------------------------------------------------
// FloatPLus/1         0.062 ns        0.062 ns   11176220936
// FloatPLus/4          2.23 ns         2.23 ns    309735377
// FloatPLus/16         4.53 ns         4.53 ns    154850523
// FloatPLus/64         16.2 ns         16.2 ns     43028154
// FloatPLus/256        62.3 ns         63.2 ns     10706858
// FloatPLus/1024        256 ns          260 ns      2761794
// loop unroll and no vectorize

BENCHMARK(DoublePLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// DoublePLus/1         0.064 ns        0.064 ns   11469351762
// DoublePLus/4          2.08 ns         2.08 ns    344543407
// DoublePLus/16         4.55 ns         4.55 ns    155128572
// DoublePLus/64         17.3 ns         17.3 ns     43104670
// DoublePLus/256        64.7 ns         64.7 ns     10906333
// DoublePLus/1024        297 ns          297 ns      2364224
// loop unroll and no vectorize

BENCHMARK(LongDoublePLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// DoublePLus/1         0.062 ns        0.062 ns   11616188519
// DoublePLus/4          2.08 ns         2.07 ns    348473392
// DoublePLus/16         4.52 ns         4.50 ns    154095096
// DoublePLus/64         16.1 ns         16.1 ns     43560609
// DoublePLus/256        63.3 ns         63.1 ns     10999072
// DoublePLus/1024        298 ns          298 ns      2356307
// loop unroll and no vectorize

// the code seems to do only loop unroll(which has to be forced) even I force the vectorize
// which make senses because how would you do? 

BENCHMARK_MAIN();