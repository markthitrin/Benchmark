#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile char c = 'a';
  }
}

static void ShortAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile short s = 69;
  }
}

static void IntAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile int i = 69;
  }
}

static void LongLongAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile long long l = 69;
  }
}

static void FloatAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile float f = 1.6;
  }
}

static void DoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile double d = 1.660;
  }
}

static void LongDoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile long double ld = 9.4444332;
  }
}

BENCHMARK(CharAssignment); 
BENCHMARK(ShortAssignment); 
BENCHMARK(IntAssignment); 
BENCHMARK(LongLongAssignment); 
BENCHMARK(FloatAssignment); 
BENCHMARK(DoubleAssignment); 
BENCHMARK(LongDoubleAssignment); 
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAssignment            0.240 ns        0.240 ns   2932931409
// ShortAssignment           0.240 ns        0.240 ns   2916479186
// IntAssignment             0.239 ns        0.239 ns   2938149473
// LongLongAssignment        0.240 ns        0.240 ns   2949315205
// FloatAssignment           0.239 ns        0.239 ns   2939481427
// DoubleAssignment          0.239 ns        0.239 ns   2941805869
// LongDoubleAssignment       3.16 ns         3.16 ns    224951395

BENCHMARK_MAIN();