#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

template<int size>
static void FloatPLus(benchmark::State& state) {
  float* a = (float*)malloc(sizeof(float) * size);
  float* b = (float*)malloc(sizeof(float) * size);
  float* c = (float*)malloc(sizeof(float) * size);
  for(auto _ : state) {
    for(int q = 0;q < size;q++) {
      c[q] = a[q] + b[q];
    }
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
}

#define BENCHMARK_TEMPATB(func) \
BENCHMARK_TEMPLATE(func,1);\
BENCHMARK_TEMPLATE(func,4);\
BENCHMARK_TEMPLATE(func,16);\
BENCHMARK_TEMPLATE(func,64);\
BENCHMARK_TEMPLATE(func,256);\
BENCHMARK_TEMPLATE(func,1024);\
BENCHMARK_TEMPLATE(func,4096);\
BENCHMARK_TEMPLATE(func,16384);\
BENCHMARK_TEMPLATE(func,65536);\
BENCHMARK_TEMPLATE(func,262144);\
BENCHMARK_TEMPLATE(func,1048576);\
BENCHMARK_TEMPLATE(func,4194304);\
BENCHMARK_TEMPLATE(func,16777216);

BENCHMARK_TEMPATB(FloatPLus);

BENCHMARK_MAIN();