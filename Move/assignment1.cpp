#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAssignment(benchmark::State& state) {
  for(auto _ : state) {
    char c = 'a';
    escape(&c);
  }
}

static void ShortAssignment(benchmark::State& state) {
  for(auto _ : state) {
    short s = 69;
    escape(&s);
  }
}

static void IntAssignment(benchmark::State& state) {
  for(auto _ : state) {
    int i = 69;
    escape(&i);
  }
}

static void LongLongAssignment(benchmark::State& state) {
  for(auto _ : state) {
    long long l = 69;
    escape(&l);
  }
}

static void FloatAssignment(benchmark::State& state) {
  for(auto _ : state) {
    float f = 1.6;
    escape(&f);
  }
}

static void DoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    double d = 1.660;
    escape(&d);
  }
}

static void LongDoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    long double ld = 9.4444332;
    escape(&ld);
  }
}

BENCHMARK(CharAssignment); // 0.476ns
// NOTE :
// The function test alone get 0.238ns
// but for some reason, testing together make the function 2 times slower, and the result seems to be consistant
// The asm code is the same (movb,sub,jmp)
// I believe that the reason is the instruction cache line. The loop might not align in the same cache line
// resulting the fecthing multiple instruction to do super scalar or anythings else become slow as it can
// not fetch the whole together.
// We can try put int = 0;escape(&u) to any function and the function run faster
// we can do multiples benchmark CharAssignment and every function run faster
// We can do CharAssignment alone and the function run faster.
BENCHMARK(ShortAssignment); // 0.238ns
BENCHMARK(IntAssignment); // 0.238ns
BENCHMARK(LongLongAssignment); // 0.238ns
BENCHMARK(FloatAssignment); // 0.238ns
BENCHMARK(DoubleAssignment); // 0.238ns
BENCHMARK(LongDoubleAssignment); // 3.17ns
// need special instruction to load and store

BENCHMARK_MAIN();