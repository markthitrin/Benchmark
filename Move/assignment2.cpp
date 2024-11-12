#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  char* a = new char[array_size];
  char* c = new char[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}

static void ShortAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  short* a = new short[array_size];
  short* c = new short[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}

static void IntAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  int* a = new int[array_size];
  int* c = new int[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}

static void LongLongAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  long long* a = new long long[array_size];
  long long* c = new long long[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}
static void FloatAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  float* a = new float[array_size];
  float* c = new float[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}

static void DoubleAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  double* a = new double[array_size];
  double* c = new double[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}

static void LongDoubleAssignment(benchmark::State& state) {
  const int array_size = state.range(0);
  long double* a = new long double[array_size];
  long double* c = new long double[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
    escape(&c);
  }
  delete a;
  delete c;
}

BENCHMARK(CharAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 0.724, 1.69, 5.73, 23.0, 96.1, 370 ns
// loop unroll 2 step
// no memcpy
BENCHMARK(ShortAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.64, 2.63, 2.64, 2.65, 4.89, 17.2 ns
// memcpy
BENCHMARK(IntAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 3.45, 2.67, 2.65, 2.68, 8.34, 31.9 ns
// memcpy
BENCHMARK(LongLongAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.64, 2.63, 2.65, 4.81, 16.0, 63.6 ns
// memcpy
BENCHMARK(FloatAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.70, 2.65, 2.65, 2.68, 8.49, 32.7 ns
// memcpy
BENCHMARK(DoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.77, 2.65, 2.69, 4.83, 16.1, 64.0 ns
// memcpy
BENCHMARK(LongDoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.67, 2.66, 4.15, 8.40, 33.8, 151 ns
// memcpy


BENCHMARK_MAIN();