#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAssignment(benchmark::State& state) {
  for(auto _ : state) {
    char c = 'a';
    benchmark::DoNotOptimize(&c);
  }
}

static void ShortAssignment(benchmark::State& state) {
  for(auto _ : state) {
    short s = 69;
    benchmark::DoNotOptimize(&s);

  }
}

static void IntAssignment(benchmark::State& state) {
  for(auto _ : state) {
    int i = 69;
    benchmark::DoNotOptimize(&i);
  }
}

static void LongLongAssignment(benchmark::State& state) {
  for(auto _ : state) {
    long long l = 69;
    benchmark::DoNotOptimize(&l);
  }
}

static void FloatAssignment(benchmark::State& state) {
  for(auto _ : state) {
    float f = 1.6;
    benchmark::DoNotOptimize(&f);
  }
}

static void DoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    double d = 1.660;
    benchmark::DoNotOptimize(&d);
  }
}

static void LongDoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    long double ld = 9.4444332;
    benchmark::DoNotOptimize(&ld);
  }
}

BENCHMARK(CharAssignment); 
BENCHMARK(ShortAssignment); 
BENCHMARK(IntAssignment); 
BENCHMARK(LongLongAssignment); 
BENCHMARK(FloatAssignment); 
BENCHMARK(DoubleAssignment); 
BENCHMARK(LongDoubleAssignment);
// g++
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAssignment            0.238 ns        0.238 ns   2935697251
// ShortAssignment           0.239 ns        0.239 ns   2924698327
// IntAssignment             0.238 ns        0.238 ns   2937756761
// LongLongAssignment        0.238 ns        0.238 ns   2938765238
// FloatAssignment           0.479 ns        0.479 ns   1461814874
// DoubleAssignment          0.238 ns        0.238 ns   2931379370
// LongDoubleAssignment       3.13 ns         3.13 ns    223506356


// clang++
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAssignment             1.19 ns         1.19 ns    585110323
// ShortAssignment            1.19 ns         1.19 ns    582488337
// IntAssignment             0.482 ns        0.482 ns   1450504447
// LongLongAssignment        0.483 ns        0.483 ns   1450459346
// FloatAssignment           0.483 ns        0.483 ns   1450967247
// DoubleAssignment          0.482 ns        0.482 ns   1452069673
// LongDoubleAssignment       1.45 ns         1.45 ns    482231638


// The result is exactly the same as escape function

BENCHMARK_MAIN();