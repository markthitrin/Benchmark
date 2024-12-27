#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void StdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  long long* arr = (long long*)malloc(sizeof(long long) * array_size);
  for(int q = 0;q < array_size;q++) {
    arr[q] = (long long)std::rand() << 32 | std::rand();
  }
  std::sort(arr + array_size - std::min(array_size,100),arr + array_size);
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
}

BENCHMARK(StdSort)->RangeMultiplier(2)->Range(1,1<<28);
// ------------------------------------------------------------
// Benchmark                  Time             CPU   Iterations
// ------------------------------------------------------------
// StdSort/1               4.10 ns         4.10 ns    170703215
// StdSort/2               4.82 ns         4.82 ns    144972539
// StdSort/4               7.01 ns         7.00 ns     99125070
// StdSort/8               9.87 ns         9.87 ns     70759981
// StdSort/16              15.8 ns         15.8 ns     44504236
// StdSort/32              51.7 ns         51.7 ns     13763561
// StdSort/64               136 ns          136 ns      5205412
// StdSort/128              338 ns          338 ns      2102341
// StdSort/256              879 ns          879 ns       794917
// StdSort/512             2095 ns         2095 ns       339411
// StdSort/1024            4439 ns         4438 ns       156825
// StdSort/2048            9761 ns         9760 ns        71578
// StdSort/4096           20920 ns        20918 ns        32917
// StdSort/8192           44839 ns        44832 ns        15588
// StdSort/16384          96555 ns        96544 ns         7278
// StdSort/32768         203513 ns       203487 ns         3430
// StdSort/65536         431067 ns       431022 ns         1605
// StdSort/131072        916962 ns       916840 ns          731
// StdSort/262144       1961531 ns      1961249 ns          346
// StdSort/524288       4421723 ns      4421078 ns          157
// StdSort/1048576     11114443 ns     11112066 ns           50
// StdSort/2097152     25571025 ns     25567217 ns           26
// StdSort/4194304     69441964 ns     69431442 ns            9
// StdSort/8388608    516475453 ns    516318660 ns            1
// StdSort/16777216  1074514681 ns   1074271326 ns            1
// StdSort/33554432  2244132821 ns   2243650259 ns            1
// StdSort/67108864  4679760733 ns   4678706543 ns            1
// StdSort/134217728 9702393346 ns   9699597086 ns            1
// StdSort/268435456 2.5103e+10 ns   2.5096e+10 ns            1

BENCHMARK_MAIN();