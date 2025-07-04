#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
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
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(FloatPLus)->RangeMultiplier(4)->Range(1, 1<<24);

BENCHMARK_MAIN();