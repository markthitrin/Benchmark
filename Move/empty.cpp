#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void Empty(benchmark::State& state) {
  for(auto _ : state) {
    escape(nullptr);
  }
}

BENCHMARK(Empty);

BENCHMARK_MAIN();