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

BENCHMARK(CharPLus)->RangeMultiplier(4)->Range(1, 1<<15);
// ---------------------------------------------------------
// Benchmark               Time             CPU   Iterations
// ---------------------------------------------------------
// CharPLus/1          0.369 ns        0.368 ns   1946628746
// CharPLus/4           1.56 ns         1.55 ns    458663124
// CharPLus/16          5.62 ns         5.60 ns    119976977
// CharPLus/64          24.6 ns         24.5 ns     28988275
// CharPLus/256          101 ns          100 ns      6840721
// CharPLus/1024         393 ns          392 ns      1800422
// CharPLus/4096        1569 ns         1566 ns       443306 
// CharPLus/16384       7332 ns         7319 ns        98321 // maybe of cache load
// CharPLus/32768      14590 ns        14564 ns        47820

BENCHMARK(ShortPLus)->RangeMultiplier(4)->Range(1, 1<<15);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// ShortPLus/1          0.359 ns        0.359 ns   1946360853
// ShortPLus/4           1.45 ns         1.45 ns    480810757
// ShortPLus/16          5.46 ns         5.46 ns    126540795
// ShortPLus/64          23.8 ns         23.8 ns     29368242
// ShortPLus/256          103 ns          103 ns      6853154
// ShortPLus/1024         393 ns          393 ns      1782473
// ShortPLus/4096        1722 ns         1721 ns       403826 
// ShortPLus/16384       7315 ns         7312 ns        95300 // maybe of cache load
// ShortPLus/32768      14863 ns        14855 ns        47198

BENCHMARK(IntPLus)->RangeMultiplier(4)->Range(1, 1<<15);
// --------------------------------------------------------
// Benchmark              Time             CPU   Iterations
// --------------------------------------------------------
// IntPLus/1          0.361 ns        0.361 ns   1937455955
// IntPLus/4           1.44 ns         1.44 ns    479480351
// IntPLus/16          5.47 ns         5.47 ns    127282348
// IntPLus/64          23.9 ns         23.9 ns     29377661
// IntPLus/256          102 ns          102 ns      6982264
// IntPLus/1024         395 ns          395 ns      1779010
// IntPLus/4096        1837 ns         1836 ns       344228 // maybe of cache load
// IntPLus/16384       7484 ns         7480 ns        93245
// IntPLus/32768      14819 ns        14811 ns        46730

BENCHMARK(LongLongPLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------
// Benchmark                  Time             CPU   Iterations
// ------------------------------------------------------------
// LongLongPLus/1         0.366 ns        0.365 ns   1912821138
// LongLongPLus/4          1.38 ns         1.38 ns    468763107
// LongLongPLus/16         5.70 ns         5.70 ns    124564072
// LongLongPLus/64         23.9 ns         23.9 ns     29409150
// LongLongPLus/256         100 ns          100 ns      6971360
// LongLongPLus/1024        441 ns          441 ns      1627180 // maybe of cache load

BENCHMARK(FloatPLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ---------------------------------------------------------
// Benchmark               Time             CPU   Iterations
// ---------------------------------------------------------
// FloatPLus/1         0.360 ns        0.360 ns   1946729245
// FloatPLus/4          1.47 ns         1.47 ns    474695093
// FloatPLus/16         4.91 ns         4.90 ns    142696178
// FloatPLus/64         19.0 ns         19.0 ns     36792686
// FloatPLus/256        80.1 ns         80.0 ns      8739123
// FloatPLus/1024        306 ns          306 ns      2292721

BENCHMARK(DoublePLus)->RangeMultiplier(4)->Range(1, 1<<10);
// ----------------------------------------------------------
// Benchmark                Time             CPU   Iterations
// ----------------------------------------------------------
// DoublePLus/1         0.374 ns        0.374 ns   1936432594
// DoublePLus/4          1.35 ns         1.34 ns    511005959
// DoublePLus/16         5.06 ns         5.05 ns    128858820
// DoublePLus/64         19.4 ns         19.4 ns     36079507
// DoublePLus/256        81.8 ns         81.7 ns      8622256
// DoublePLus/1024        338 ns          338 ns      2171041 // probably of cache load

BENCHMARK(LongDoublePLus)->RangeMultiplier(4)->Range(1, 1<<10);
// --------------------------------------------------------------
// Benchmark                    Time             CPU   Iterations
// --------------------------------------------------------------
// LongDoublePLus/1         0.476 ns        0.477 ns   1469015115
// LongDoublePLus/4          9.33 ns         9.35 ns     74629718
// LongDoublePLus/16         46.6 ns         46.7 ns     14986792
// LongDoublePLus/64          196 ns          196 ns      3568722
// LongDoublePLus/256         799 ns          800 ns       873330
// LongDoublePLus/1024       3187 ns         3191 ns       219412

// normal add no unroll and no vectorize. for vectorize and unroll force see plus2 and plus3
// float,double seems to be faster than int,short,long long
// some specific size of array size seems to face cache load delay
// 3 instructions for add and 2 instructions for loop

BENCHMARK_MAIN();