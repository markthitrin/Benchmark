#include <benchmark/benchmark.h>
#include <cstring>

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
    std::memcpy(c,a,array_size * sizeof(char));
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
    std::memcpy(c,a,array_size * sizeof(short));
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
    std::memcpy(c,a,array_size * sizeof(int));
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
    std::memcpy(c,a,array_size * sizeof(long long));
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
    std::memcpy(c,a,array_size * sizeof(float));
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
    std::memcpy(c,a,array_size * sizeof(double));
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
    std::memcpy(c,a,array_size * sizeof(long double));
    escape(&c);
  }
  delete a;
  delete c;
}

BENCHMARK(CharAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.89, 2.65, 2.63, 2.64, 2.66, 8.28 ns
BENCHMARK(ShortAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.92, 2.69, 2.65, 2.26, 2.69, 10.2 ns
BENCHMARK(IntAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.65, 2.64, 2.22, 2.67, 8.34, 31.7 ns
BENCHMARK(LongLongAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.44, 2.45, 3.58, 4.82, 18.9, 64.2 ns
BENCHMARK(FloatAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.72, 2.71, 2.45, 2.69, 8.34, 31.8 ns
BENCHMARK(DoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.47, 2.09, 2.46, 5.08, 16.3, 65.0 ns
BENCHMARK(LongDoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<10); // 2.63, 2.17, 4.09, 11.4, 32.9, 150 ns
// memcpy with 512 bytes


BENCHMARK_MAIN();