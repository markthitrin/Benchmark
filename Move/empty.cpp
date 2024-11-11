#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

// 0.119ns
// There are 3 instructions 
//  sub - interate loop
//  cmp
//  jmp
static void Empty(benchmark::State& state) {
  for(auto _ : state) {
    escape(nullptr);
  }
}

BENCHMARK(Empty);

BENCHMARK_MAIN();