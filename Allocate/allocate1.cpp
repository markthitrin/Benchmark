#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = new char[array_size];
    escape(p);
    delete p;
  }
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    short* p = new short[array_size];
    escape(p);
    delete p;
  }
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    int* p = new int[array_size];
    escape(p);
    delete p;
  }
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long long* p = new long long[array_size];
    escape(p);
    delete p;
  }
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    float* p = new float[array_size];
    escape(p);
    delete p;
  }
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    double* p = new double[array_size];
    escape(p);
    delete p;
  }
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long double* p = new long double[array_size];
    escape(p);
    delete p;
  }
}

static void Allocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = new char[array_size];
    escape(p);
    delete p;
  }
}

BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.2, 12.1, 11.8, 11.8, 11.8, 11.8 ns
BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.5, 12.5, 11.9, 11.9, 12.0, 40.5 ns
BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.4, 12.2, 12.2, 12.1, 12.2, 30.7 ns
BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.3, 12.1, 12.1, 12.2, 41.5, 32.7 ns 
BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.0, 11.8, 11.8, 11.8, 12.1, 31.4 ns
BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.5, 12.0, 12.1, 41.8, 31.0 ns
BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<10); // 12.1, 12.2, 12.4, 12.5, 33.1, 31.9

// The big datea type need memory allignment

BENCHMARK(Allocate)->RangeMultiplier(2)->Range(1,1<<30);
// --------------------------------------------------------------
// Benchmark                    Time             CPU   Iterations
// --------------------------------------------------------------
// Allocate/1                11.9 ns         11.9 ns     58755253
// Allocate/2                11.9 ns         11.9 ns     58196377
// Allocate/4                11.9 ns         11.9 ns     58454606
// Allocate/8                11.9 ns         11.9 ns     58358757
// Allocate/16               11.7 ns         11.7 ns     59647536
// Allocate/32               11.7 ns         11.7 ns     59830277
// Allocate/64               11.7 ns         11.7 ns     59527152
// Allocate/128              11.7 ns         11.7 ns     59775970
// Allocate/256              11.7 ns         11.7 ns     59673123
// Allocate/512              12.1 ns         12.0 ns     59494636
// Allocate/1024             12.0 ns         12.0 ns     58537633 2 ^ 10
// Allocate/2048             31.4 ns         31.4 ns     23032563 2 ^ 11
// Allocate/4096             31.6 ns         31.6 ns     22308140
// Allocate/8192             32.2 ns         32.1 ns     22860499
// Allocate/16384            31.0 ns         31.0 ns     22898892
// Allocate/32768            32.9 ns         32.9 ns     21591230
// Allocate/65536            32.8 ns         32.8 ns     20954641
// Allocate/131072           35.8 ns         35.8 ns     19719053
// Allocate/262144           32.9 ns         32.8 ns     21580616
// Allocate/524288           32.3 ns         32.3 ns     21122638
// Allocate/1048576          31.5 ns         31.5 ns     21708349
// Allocate/2097152          33.8 ns         33.7 ns     22825431
// Allocate/4194304          32.7 ns         32.7 ns     20632530
// Allocate/8388608          32.5 ns         32.5 ns     20944292
// Allocate/16777216         32.4 ns         32.4 ns     21675762 2 ^ 24
// Allocate/33554432         9508 ns         9215 ns        73138 2 ^ 25
// Allocate/67108864        11095 ns        10680 ns        65579 2 ^ 26
// Allocate/134217728       12203 ns        11791 ns        59516 2 ^ 27
// Allocate/268435456       14275 ns        13855 ns        49985 2 ^ 28
// Allocate/536870912       18584 ns        18084 ns        40806 2 ^ 29
// Allocate/1073741824      25365 ns        24814 ns        28343 2 ^ 30

BENCHMARK_MAIN();