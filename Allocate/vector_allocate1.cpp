#include <benchmark/benchmark.h>
#include <vector>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<char> v(array_size);
    escape(v.data());
  }
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<short> v(array_size);
    escape(v.data());
  }
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<int> v(array_size);
    escape(v.data());
  }
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long long> v(array_size);
    escape(v.data());
  }
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<float> v(array_size);
    escape(v.data());
  }
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<double> v(array_size);
    escape(v.data());
  }
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long double> v(array_size);
    escape(v.data());
  }
}

BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAllocate/1             11.2 ns         11.2 ns     62363389
// CharAllocate/4             13.5 ns         13.4 ns     50861150
// CharAllocate/16            12.6 ns         12.6 ns     53977172
// CharAllocate/64            12.4 ns         12.4 ns     56567210
// CharAllocate/256           12.8 ns         12.8 ns     54547610
// CharAllocate/1024          17.0 ns         17.0 ns     41168153
// CharAllocate/4096          49.6 ns         49.6 ns     14050693
// CharAllocate/16384          151 ns          150 ns      4676320
// CharAllocate/65536          531 ns          531 ns      1322891
// CharAllocate/262144        2041 ns         2039 ns       342145
// CharAllocate/1048576      11787 ns        11777 ns        59337

// BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<20); 
// BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20); 
// BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20); 

BENCHMARK_MAIN();