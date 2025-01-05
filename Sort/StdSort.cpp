#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  char* arr = (char*)malloc(sizeof(char) * array_size);
  for(int q = 0;q < array_size;q++) {
    arr[q] = (char)std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
  free(arr);
}

static void ShortStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  short* arr = (short*)malloc(sizeof(short) * array_size);
  for(int q = 0;q < array_size;q++) {
    arr[q] = (short)std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
  free(arr);
}

static void IntStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  int* arr = (int*)malloc(sizeof(int) * array_size);
  for(int q = 0;q < array_size;q++) {
    arr[q] = (int)std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
  free(arr);
}

static void LongLongStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  long long* arr = (long long*)malloc(sizeof(long long) * array_size);
  for(int q = 0;q < array_size;q++) {
    arr[q] = (long long)std::rand() << 32 | std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
  free(arr);
}

static void FloatStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  float* arr = (float*)malloc(sizeof(float) * array_size);
  int* ptr = reinterpret_cast<int*>(arr);
  for(int q = 0;q < array_size;q++) {
    ptr[q] = std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
  free(arr);
}

static void DoubleStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  double* arr = (double*)malloc(sizeof(double) * array_size);
  long long* ptr = reinterpret_cast<long long*>(arr);
  for(int q = 0;q < array_size;q++) {
    ptr[q] = (long long)std::rand() << 32 | std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
  free(arr);
}

static void LongDoubleStdSort(benchmark::State& state) {
  const int array_size = state.range(0);
  long double* arr = (long double*)malloc(sizeof(long double) * array_size);
  long long* ptr = reinterpret_cast<long long*>(arr);
  for(int q = 0;q < array_size * 2;q++) {
    ptr[q] = (long long)std::rand() << 32 | std::rand();
  }
  for(auto _ : state) {
    std::sort(arr,arr + array_size);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
  free(arr);
}

BENCHMARK(CharStdSort)->RangeMultiplier(4)->Range(1,1<<26);
BENCHMARK(ShortStdSort)->RangeMultiplier(4)->Range(1,1<<26);
BENCHMARK(IntStdSort)->RangeMultiplier(4)->Range(1,1<<26);
BENCHMARK(LongLongStdSort)->RangeMultiplier(4)->Range(1,1<<26);
BENCHMARK(FloatStdSort)->RangeMultiplier(4)->Range(1,1<<26);
BENCHMARK(DoubleStdSort)->RangeMultiplier(4)->Range(1,1<<26);
BENCHMARK(LongDoubleStdSort)->RangeMultiplier(4)->Range(1,1<<26);

// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharStdSort/1                    4.75 ns         4.75 ns    146415482 bytes_per_second=200.975Mi/s items_per_second=210.737M/s
// CharStdSort/4                    7.03 ns         7.03 ns     99317773 bytes_per_second=542.807Mi/s items_per_second=142.294M/s
// CharStdSort/16                   15.8 ns         15.8 ns     44425458 bytes_per_second=968.224Mi/s items_per_second=63.4536M/s
// CharStdSort/64                    135 ns          135 ns      4804881 bytes_per_second=452.19Mi/s items_per_second=7.40868M/s
// CharStdSort/256                   912 ns          912 ns       766670 bytes_per_second=267.837Mi/s items_per_second=1.09706M/s
// CharStdSort/1024                 4585 ns         4585 ns       149272 bytes_per_second=213.013Mi/s items_per_second=218.125k/s
// CharStdSort/4096                22439 ns        22437 ns        30700 bytes_per_second=174.096Mi/s items_per_second=44.5685k/s
// CharStdSort/16384               97916 ns        97901 ns         7110 bytes_per_second=159.6Mi/s items_per_second=10.2144k/s
// CharStdSort/65536              396962 ns       396908 ns         1781 bytes_per_second=157.467Mi/s items_per_second=2.51947k/s
// CharStdSort/262144            1712728 ns      1712432 ns          396 bytes_per_second=145.991Mi/s items_per_second=583.965/s
// CharStdSort/1048576           7991347 ns      7990012 ns           70 bytes_per_second=125.156Mi/s items_per_second=125.156/s
// CharStdSort/4194304          41582754 ns     41573800 ns           14 bytes_per_second=96.2144Mi/s items_per_second=24.0536/s
// CharStdSort/16777216        514020174 ns    513946955 ns            1 bytes_per_second=31.1316Mi/s items_per_second=1.94573/s
// CharStdSort/67108864       2114840318 ns   2114376958 ns            1 bytes_per_second=30.269Mi/s items_per_second=0.472953/s

// ShortStdSort/1                   4.49 ns         4.49 ns    158690147 bytes_per_second=424.794Mi/s items_per_second=222.714M/s
// ShortStdSort/4                   6.88 ns         6.88 ns    100120519 bytes_per_second=1.0832Gi/s items_per_second=145.385M/s
// ShortStdSort/16                  15.6 ns         15.6 ns     44615120 bytes_per_second=1.91087Gi/s items_per_second=64.1182M/s
// ShortStdSort/64                   145 ns          145 ns      4798380 bytes_per_second=844.519Mi/s items_per_second=6.9183M/s
// ShortStdSort/256                  996 ns          995 ns       697953 bytes_per_second=490.531Mi/s items_per_second=1.00461M/s
// ShortStdSort/1024                5283 ns         5282 ns       131411 bytes_per_second=369.764Mi/s items_per_second=189.319k/s
// ShortStdSort/4096               25877 ns        25871 ns        27485 bytes_per_second=301.98Mi/s items_per_second=38.6534k/s
// ShortStdSort/16384             119379 ns       119369 ns         5847 bytes_per_second=261.792Mi/s items_per_second=8.37736k/s
// ShortStdSort/65536             589745 ns       589666 ns         1165 bytes_per_second=211.984Mi/s items_per_second=1.69587k/s
// ShortStdSort/262144           3009196 ns      3008842 ns          232 bytes_per_second=166.177Mi/s items_per_second=332.354/s
// ShortStdSort/1048576         12325279 ns     12323788 ns           48 bytes_per_second=162.288Mi/s items_per_second=81.1439/s
// ShortStdSort/4194304         58251350 ns     58240133 ns           10 bytes_per_second=137.362Mi/s items_per_second=17.1703/s
// ShortStdSort/16777216       853722295 ns    853649594 ns            1 bytes_per_second=37.4861Mi/s items_per_second=1.17144/s
// ShortStdSort/67108864      3331331484 ns   3330741430 ns            1 bytes_per_second=38.4299Mi/s items_per_second=0.300233/s

// IntStdSort/1                     4.66 ns         4.66 ns    151928958 bytes_per_second=819.359Mi/s items_per_second=214.79M/s
// IntStdSort/4                     7.07 ns         7.07 ns     98395913 bytes_per_second=2.10735Gi/s items_per_second=141.422M/s
// IntStdSort/16                    15.9 ns         15.9 ns     44068220 bytes_per_second=3.75066Gi/s items_per_second=62.9256M/s
// IntStdSort/64                     153 ns          153 ns      4620541 bytes_per_second=1.55706Gi/s items_per_second=6.53077M/s
// IntStdSort/256                   1013 ns         1012 ns       681855 bytes_per_second=964.614Mi/s items_per_second=987.764k/s
// IntStdSort/1024                  5269 ns         5267 ns       130947 bytes_per_second=741.583Mi/s items_per_second=189.845k/s
// IntStdSort/4096                 25725 ns        25723 ns        27609 bytes_per_second=607.427Mi/s items_per_second=38.8753k/s
// IntStdSort/16384               117122 ns       117108 ns         5908 bytes_per_second=533.694Mi/s items_per_second=8.53911k/s
// IntStdSort/65536               530537 ns       530455 ns         1248 bytes_per_second=471.294Mi/s items_per_second=1.88518k/s
// IntStdSort/262144             2440340 ns      2439813 ns          282 bytes_per_second=409.867Mi/s items_per_second=409.867/s
// IntStdSort/1048576           13300861 ns     13297320 ns           49 bytes_per_second=300.813Mi/s items_per_second=75.2031/s
// IntStdSort/4194304           91583351 ns     91575890 ns            6 bytes_per_second=174.718Mi/s items_per_second=10.9199/s
// IntStdSort/16777216        1102980395 ns   1102788018 ns            1 bytes_per_second=58.0347Mi/s items_per_second=0.906793/s
// IntStdSort/67108864        4701249982 ns   4699935412 ns            1 bytes_per_second=54.4688Mi/s items_per_second=0.212769/s

// LongLongStdSort/1                4.45 ns         4.45 ns    159941511 bytes_per_second=1.67486Gi/s items_per_second=224.796M/s
// LongLongStdSort/4                6.84 ns         6.83 ns    100700845 bytes_per_second=4.36089Gi/s items_per_second=146.327M/s
// LongLongStdSort/16               15.8 ns         15.8 ns     44423849 bytes_per_second=7.5579Gi/s items_per_second=63.4002M/s
// LongLongStdSort/64                146 ns          146 ns      4761550 bytes_per_second=3.27166Gi/s items_per_second=6.86117M/s
// LongLongStdSort/256               917 ns          917 ns       747956 bytes_per_second=2.08087Gi/s items_per_second=1.09098M/s
// LongLongStdSort/1024             4697 ns         4696 ns       147560 bytes_per_second=1.62475Gi/s items_per_second=212.959k/s
// LongLongStdSort/4096            22125 ns        22122 ns        32143 bytes_per_second=1.37953Gi/s items_per_second=45.2043k/s
// LongLongStdSort/16384           98629 ns        98621 ns         6981 bytes_per_second=1.23777Gi/s items_per_second=10.1398k/s
// LongLongStdSort/65536          458764 ns       458733 ns         1588 bytes_per_second=1.06441Gi/s items_per_second=2.17992k/s
// LongLongStdSort/262144        2379083 ns      2378666 ns          317 bytes_per_second=840.808Mi/s items_per_second=420.404/s
// LongLongStdSort/1048576      13712857 ns     13709463 ns           49 bytes_per_second=583.539Mi/s items_per_second=72.9423/s
// LongLongStdSort/4194304     100844353 ns    100817611 ns            5 bytes_per_second=317.405Mi/s items_per_second=9.9189/s
// LongLongStdSort/16777216   1206434168 ns   1206151647 ns            1 bytes_per_second=106.123Mi/s items_per_second=0.829083/s
// LongLongStdSort/67108864   4736779458 ns   4735266060 ns            1 bytes_per_second=108.125Mi/s items_per_second=0.211181/s

// FloatStdSort/1                   4.04 ns         4.04 ns    172305611 bytes_per_second=943.305Mi/s items_per_second=247.282M/s
// FloatStdSort/4                   6.84 ns         6.84 ns    101960018 bytes_per_second=2.17945Gi/s items_per_second=146.26M/s
// FloatStdSort/16                  15.7 ns         15.7 ns     44645573 bytes_per_second=3.78907Gi/s items_per_second=63.57M/s
// FloatStdSort/64                   152 ns          152 ns      4585669 bytes_per_second=1.56871Gi/s items_per_second=6.57965M/s
// FloatStdSort/256                 3395 ns         3394 ns       208320 bytes_per_second=287.692Mi/s items_per_second=294.596k/s
// FloatStdSort/1024               31302 ns        31297 ns        76983 bytes_per_second=124.813Mi/s items_per_second=31.9521k/s
// FloatStdSort/4096              195991 ns       195949 ns         3911 bytes_per_second=79.7403Mi/s items_per_second=5.10338k/s
// FloatStdSort/16384             802949 ns       802773 ns          858 bytes_per_second=77.8551Mi/s items_per_second=1.24568k/s
// FloatStdSort/65536            3409472 ns      3409134 ns          208 bytes_per_second=73.3324Mi/s items_per_second=293.33/s
// FloatStdSort/262144          14668189 ns     14667113 ns           49 bytes_per_second=68.1797Mi/s items_per_second=68.1797/s
// FloatStdSort/1048576         63503088 ns     63490883 ns           10 bytes_per_second=63.0012Mi/s items_per_second=15.7503/s
// FloatStdSort/4194304        349045622 ns    348971251 ns            2 bytes_per_second=45.849Mi/s items_per_second=2.86557/s
// FloatStdSort/16777216      1371191187 ns   1370973402 ns            1 bytes_per_second=46.6822Mi/s items_per_second=0.729409/s
// FloatStdSort/67108864      5847931596 ns   5846254090 ns            1 bytes_per_second=43.7887Mi/s items_per_second=0.17105/s

// DoubleStdSort/1                  3.68 ns         3.68 ns    191700618 bytes_per_second=2.02341Gi/s items_per_second=271.577M/s
// DoubleStdSort/4                  6.63 ns         6.63 ns    102226633 bytes_per_second=4.49497Gi/s items_per_second=150.826M/s
// DoubleStdSort/16                 15.8 ns         15.8 ns     43322232 bytes_per_second=7.56053Gi/s items_per_second=63.4223M/s
// DoubleStdSort/64                  156 ns          156 ns      4458768 bytes_per_second=3.06291Gi/s items_per_second=6.42338M/s
// DoubleStdSort/256                1026 ns         1026 ns       683043 bytes_per_second=1.85859Gi/s items_per_second=974.439k/s
// DoubleStdSort/1024               4793 ns         4793 ns       133844 bytes_per_second=1.59194Gi/s items_per_second=208.659k/s
// DoubleStdSort/4096             203451 ns       203422 ns        10000 bytes_per_second=153.622Mi/s items_per_second=4.9159k/s
// DoubleStdSort/16384            849058 ns       848876 ns          804 bytes_per_second=147.254Mi/s items_per_second=1.17803k/s
// DoubleStdSort/65536           3643026 ns      3642638 ns          195 bytes_per_second=137.263Mi/s items_per_second=274.526/s
// DoubleStdSort/262144         15801619 ns     15798982 ns           52 bytes_per_second=126.59Mi/s items_per_second=63.2952/s
// DoubleStdSort/1048576       101644472 ns    101618039 ns            9 bytes_per_second=78.7262Mi/s items_per_second=9.84077/s
// DoubleStdSort/4194304       281441323 ns    281410630 ns            2 bytes_per_second=113.713Mi/s items_per_second=3.55353/s
// DoubleStdSort/16777216     1466726882 ns   1466507229 ns            1 bytes_per_second=87.2822Mi/s items_per_second=0.681892/s
// DoubleStdSort/67108864     5844833329 ns   5843001244 ns            1 bytes_per_second=87.6262Mi/s items_per_second=0.171145/s

// LongDoubleStdSort/1              3.93 ns         3.93 ns    179513889 bytes_per_second=3.78989Gi/s items_per_second=254.335M/s
// LongDoubleStdSort/4              15.9 ns         15.9 ns     44058128 bytes_per_second=3.75966Gi/s items_per_second=63.0766M/s
// LongDoubleStdSort/16             65.7 ns         65.7 ns     10590731 bytes_per_second=3.63022Gi/s items_per_second=15.2263M/s
// LongDoubleStdSort/64              600 ns          600 ns      1159214 bytes_per_second=1.5893Gi/s items_per_second=1.6665M/s
// LongDoubleStdSort/256            3911 ns         3911 ns       180635 bytes_per_second=998.896Mi/s items_per_second=255.717k/s
// LongDoubleStdSort/1024          21300 ns        21294 ns        32885 bytes_per_second=733.783Mi/s items_per_second=46.9621k/s
// LongDoubleStdSort/4096         108089 ns       108080 ns         6462 bytes_per_second=578.274Mi/s items_per_second=9.25238k/s
// LongDoubleStdSort/16384        523118 ns       523070 ns         1329 bytes_per_second=477.948Mi/s items_per_second=1.91179k/s
// LongDoubleStdSort/65536       2516319 ns      2515877 ns          284 bytes_per_second=397.476Mi/s items_per_second=397.476/s
// LongDoubleStdSort/262144     12534404 ns     12532367 ns           57 bytes_per_second=319.174Mi/s items_per_second=79.7934/s
// LongDoubleStdSort/1048576    56849566 ns     56840836 ns           12 bytes_per_second=281.488Mi/s items_per_second=17.593/s
// LongDoubleStdSort/4194304   299886578 ns    299825237 ns            3 bytes_per_second=213.458Mi/s items_per_second=3.33528/s
// LongDoubleStdSort/16777216 1257732187 ns   1257517826 ns            1 bytes_per_second=203.576Mi/s items_per_second=0.795217/s
// LongDoubleStdSort/67108864 4628039115 ns   4626566800 ns            1 bytes_per_second=221.33Mi/s items_per_second=0.216143/s
// IntStdSort
//  22.29 │1f0:   bsr      -0x70(%rbp),%r12                                                                                                                                  ▒
//   0.46 │       movslq   %r12d,%r12                                                                                                                                        ▒
//   0.50 │       add      %r12,%r12                                                                                                                                         ▒
//   0.49 │1fb:   mov      %r12,%rdx                                                                                                                                         ▒
//   1.17 │       mov      %r15,%rsi                                                                                                                                         ◆
//   0.49 │       mov      %r13,%rdi                                                                                                                                         ▒
//   1.03 │     → call     void std::__introsort_loop<int*, long, __gnu_cxx::__ops::_Iter_less_iter>(int*, int*, long, __gnu_cxx::__ops::_Iter_less_iter) [clone .isra.0]    ▒
//   0.83 │       cmp      $0x40,%r14


// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharStdSort/1                    4.25 ns         4.25 ns    163175957 bytes_per_second=224.146Mi/s items_per_second=235.035M/s
// CharStdSort/4                    5.51 ns         5.51 ns    126405813 bytes_per_second=692.25Mi/s items_per_second=181.469M/s
// CharStdSort/16                   14.2 ns         14.2 ns     48469768 bytes_per_second=1.04673Gi/s items_per_second=70.2449M/s
// CharStdSort/64                    129 ns          129 ns      5263194 bytes_per_second=472.926Mi/s items_per_second=7.74842M/s
// CharStdSort/256                   870 ns          870 ns       710139 bytes_per_second=280.726Mi/s items_per_second=1.14985M/s
// CharStdSort/1024                 4338 ns         4339 ns       152361 bytes_per_second=225.077Mi/s items_per_second=230.479k/s
// CharStdSort/4096                18953 ns        18956 ns        35970 bytes_per_second=206.074Mi/s items_per_second=52.755k/s
// CharStdSort/16384               95300 ns        95315 ns         7244 bytes_per_second=163.931Mi/s items_per_second=10.4916k/s
// CharStdSort/65536              394559 ns       394630 ns         1778 bytes_per_second=158.376Mi/s items_per_second=2.53402k/s
// CharStdSort/262144            1690881 ns      1691053 ns          396 bytes_per_second=147.837Mi/s items_per_second=591.348/s
// CharStdSort/1048576           7902916 ns      7904213 ns           70 bytes_per_second=126.515Mi/s items_per_second=126.515/s
// CharStdSort/4194304          41183369 ns     41186218 ns           13 bytes_per_second=97.1199Mi/s items_per_second=24.28/s
// CharStdSort/16777216        527651004 ns    527731019 ns            1 bytes_per_second=30.3185Mi/s items_per_second=1.8949/s
// CharStdSort/67108864       2167963423 ns   2168049303 ns            1 bytes_per_second=29.5196Mi/s items_per_second=0.461244/s

// ShortStdSort/1                   4.29 ns         4.29 ns    163005156 bytes_per_second=444.337Mi/s items_per_second=232.961M/s
// ShortStdSort/4                   6.13 ns         6.13 ns    112216112 bytes_per_second=1.21446Gi/s items_per_second=163.002M/s
// ShortStdSort/16                  15.0 ns         15.0 ns     46583761 bytes_per_second=1.99163Gi/s items_per_second=66.828M/s
// ShortStdSort/64                   131 ns          131 ns      5365535 bytes_per_second=931.492Mi/s items_per_second=7.63078M/s
// ShortStdSort/256                  877 ns          877 ns       803723 bytes_per_second=556.449Mi/s items_per_second=1.13961M/s
// ShortStdSort/1024                4468 ns         4469 ns       157144 bytes_per_second=437.068Mi/s items_per_second=223.779k/s
// ShortStdSort/4096               21432 ns        21433 ns        32837 bytes_per_second=364.516Mi/s items_per_second=46.658k/s
// ShortStdSort/16384              99473 ns        99484 ns         6987 bytes_per_second=314.121Mi/s items_per_second=10.0519k/s
// ShortStdSort/65536             500753 ns       500776 ns         1360 bytes_per_second=249.613Mi/s items_per_second=1.9969k/s
// ShortStdSort/262144           2605509 ns      2605752 ns          259 bytes_per_second=191.883Mi/s items_per_second=383.766/s
// ShortStdSort/1048576         10859452 ns     10860495 ns           52 bytes_per_second=184.154Mi/s items_per_second=92.0768/s
// ShortStdSort/4194304         52586229 ns     52587688 ns           11 bytes_per_second=152.127Mi/s items_per_second=19.0159/s
// ShortStdSort/16777216       865106195 ns    865087977 ns            1 bytes_per_second=36.9905Mi/s items_per_second=1.15595/s
// ShortStdSort/67108864      3437215736 ns   3437102699 ns            1 bytes_per_second=37.2407Mi/s items_per_second=0.290943/s

// IntStdSort/1                     4.29 ns         4.29 ns    162981702 bytes_per_second=888.532Mi/s items_per_second=232.923M/s
// IntStdSort/4                     5.91 ns         5.91 ns    120503645 bytes_per_second=2.52082Gi/s items_per_second=169.17M/s
// IntStdSort/16                    12.8 ns         12.8 ns     54330839 bytes_per_second=4.65086Gi/s items_per_second=78.0285M/s
// IntStdSort/64                     131 ns          131 ns      5254884 bytes_per_second=1.82099Gi/s items_per_second=7.63779M/s
// IntStdSort/256                    890 ns          890 ns       785540 bytes_per_second=1.07124Gi/s items_per_second=1.12327M/s
// IntStdSort/1024                  4497 ns         4497 ns       155080 bytes_per_second=868.571Mi/s items_per_second=222.354k/s
// IntStdSort/4096                 21156 ns        21158 ns        32906 bytes_per_second=738.494Mi/s items_per_second=47.2636k/s
// IntStdSort/16384                96763 ns        96773 ns         7193 bytes_per_second=645.843Mi/s items_per_second=10.3335k/s
// IntStdSort/65536               448007 ns       448051 ns         1545 bytes_per_second=557.972Mi/s items_per_second=2.23189k/s
// IntStdSort/262144             2035426 ns      2035447 ns          331 bytes_per_second=491.293Mi/s items_per_second=491.293/s
// IntStdSort/1048576            9861207 ns      9862071 ns           54 bytes_per_second=405.594Mi/s items_per_second=101.399/s
// IntStdSort/4194304           65899320 ns     65899516 ns            9 bytes_per_second=242.794Mi/s items_per_second=15.1746/s
// IntStdSort/16777216        1143564435 ns   1143500664 ns            1 bytes_per_second=55.9685Mi/s items_per_second=0.874508/s
// IntStdSort/67108864        4898202551 ns   4898046426 ns            1 bytes_per_second=52.2657Mi/s items_per_second=0.204163/s

// LongLongStdSort/1                4.30 ns         4.30 ns    162971723 bytes_per_second=1.73434Gi/s items_per_second=232.779M/s
// LongLongStdSort/4                5.92 ns         5.92 ns    119401205 bytes_per_second=5.03368Gi/s items_per_second=168.902M/s
// LongLongStdSort/16               13.8 ns         13.8 ns     50654085 bytes_per_second=8.62815Gi/s items_per_second=72.3781M/s
// LongLongStdSort/64                128 ns          128 ns      5312856 bytes_per_second=3.73748Gi/s items_per_second=7.83807M/s
// LongLongStdSort/256               879 ns          879 ns       792613 bytes_per_second=2.17043Gi/s items_per_second=1.13793M/s
// LongLongStdSort/1024             4553 ns         4554 ns       153433 bytes_per_second=1.67544Gi/s items_per_second=219.603k/s
// LongLongStdSort/4096            21718 ns        21717 ns        31747 bytes_per_second=1.40523Gi/s items_per_second=46.0467k/s
// LongLongStdSort/16384          101231 ns       101234 ns         6908 bytes_per_second=1.20582Gi/s items_per_second=9.87807k/s
// LongLongStdSort/65536          461960 ns       461969 ns         1287 bytes_per_second=1.05696Gi/s items_per_second=2.16465k/s
// LongLongStdSort/262144        2083697 ns      2083814 ns          322 bytes_per_second=959.778Mi/s items_per_second=479.889/s
// LongLongStdSort/1048576      11855706 ns     11854724 ns           48 bytes_per_second=674.836Mi/s items_per_second=84.3546/s
// LongLongStdSort/4194304     119165243 ns    119171160 ns            5 bytes_per_second=268.521Mi/s items_per_second=8.39129/s
// LongLongStdSort/16777216   1279148632 ns   1279249362 ns            1 bytes_per_second=100.059Mi/s items_per_second=0.781708/s
// LongLongStdSort/67108864   4907994573 ns   4907443811 ns            1 bytes_per_second=104.331Mi/s items_per_second=0.203772/s

// FloatStdSort/1                   4.35 ns         4.35 ns    162749524 bytes_per_second=877.543Mi/s items_per_second=230.043M/s
// FloatStdSort/4                   6.24 ns         6.23 ns    112145272 bytes_per_second=2.39257Gi/s items_per_second=160.563M/s
// FloatStdSort/16                  14.7 ns         14.7 ns     47631386 bytes_per_second=4.05197Gi/s items_per_second=67.9808M/s
// FloatStdSort/64                   162 ns          162 ns      4194704 bytes_per_second=1.47273Gi/s items_per_second=6.17708M/s
// FloatStdSort/256                 1066 ns         1065 ns       647651 bytes_per_second=917.323Mi/s items_per_second=939.339k/s
// FloatStdSort/1024               24690 ns        24649 ns        32053 bytes_per_second=158.476Mi/s items_per_second=40.5698k/s
// FloatStdSort/4096              198689 ns       198323 ns         3965 bytes_per_second=78.7854Mi/s items_per_second=5.04227k/s
// FloatStdSort/16384             858657 ns       857258 ns          803 bytes_per_second=72.9068Mi/s items_per_second=1.16651k/s
// FloatStdSort/65536            3567463 ns      3561977 ns          200 bytes_per_second=70.1857Mi/s items_per_second=280.743/s
// FloatStdSort/262144          14887320 ns     14865436 ns           48 bytes_per_second=67.2701Mi/s items_per_second=67.2701/s
// FloatStdSort/1048576         59355727 ns     59264458 ns            9 bytes_per_second=67.4941Mi/s items_per_second=16.8735/s
// FloatStdSort/4194304        355862075 ns    355292142 ns            2 bytes_per_second=45.0334Mi/s items_per_second=2.81459/s
// FloatStdSort/16777216      1417896179 ns   1415737326 ns            1 bytes_per_second=45.2061Mi/s items_per_second=0.706346/s
// FloatStdSort/67108864      6104677619 ns   6096288314 ns            1 bytes_per_second=41.9928Mi/s items_per_second=0.164034/s

// DoubleStdSort/1                  4.36 ns         4.36 ns    162044786 bytes_per_second=1.70978Gi/s items_per_second=229.483M/s
// DoubleStdSort/4                  6.21 ns         6.20 ns    111825207 bytes_per_second=4.80568Gi/s items_per_second=161.252M/s
// DoubleStdSort/16                 15.0 ns         14.9 ns     46941383 bytes_per_second=7.97831Gi/s items_per_second=66.927M/s
// DoubleStdSort/64                  145 ns          145 ns      4837222 bytes_per_second=3.28245Gi/s items_per_second=6.8838M/s
// DoubleStdSort/256                1059 ns         1058 ns       661091 bytes_per_second=1.80202Gi/s items_per_second=944.776k/s
// DoubleStdSort/1024               5336 ns         5331 ns       127996 bytes_per_second=1.43126Gi/s items_per_second=187.598k/s
// DoubleStdSort/4096             197794 ns       197608 ns         3561 bytes_per_second=158.141Mi/s items_per_second=5.06052k/s
// DoubleStdSort/16384            827088 ns       826148 ns          838 bytes_per_second=151.305Mi/s items_per_second=1.21044k/s
// DoubleStdSort/65536           3542964 ns      3539509 ns          204 bytes_per_second=141.263Mi/s items_per_second=282.525/s
// DoubleStdSort/262144         15069853 ns     15054722 ns           49 bytes_per_second=132.849Mi/s items_per_second=66.4243/s
// DoubleStdSort/1048576        86314419 ns     86236501 ns            7 bytes_per_second=92.7681Mi/s items_per_second=11.596/s
// DoubleStdSort/4194304       431701515 ns    431334094 ns            2 bytes_per_second=74.1884Mi/s items_per_second=2.31839/s
// DoubleStdSort/16777216     1477497371 ns   1476215014 ns            1 bytes_per_second=86.7082Mi/s items_per_second=0.677408/s
// DoubleStdSort/67108864     6170458418 ns   6165070077 ns            1 bytes_per_second=83.0485Mi/s items_per_second=0.162204/s

// LongDoubleStdSort/1              4.51 ns         4.50 ns    158988228 bytes_per_second=3.30825Gi/s items_per_second=222.013M/s
// LongDoubleStdSort/4              16.9 ns         16.8 ns     41341629 bytes_per_second=3.5396Gi/s items_per_second=59.3846M/s
// LongDoubleStdSort/16             66.7 ns         66.7 ns     10444481 bytes_per_second=3.57519Gi/s items_per_second=14.9954M/s
// LongDoubleStdSort/64              599 ns          599 ns      1162030 bytes_per_second=1.592Gi/s items_per_second=1.66933M/s
// LongDoubleStdSort/256            3901 ns         3899 ns       180466 bytes_per_second=1001.94Mi/s items_per_second=256.495k/s
// LongDoubleStdSort/1024          21404 ns        21389 ns        32789 bytes_per_second=730.526Mi/s items_per_second=46.7537k/s
// LongDoubleStdSort/4096         108477 ns       108413 ns         6432 bytes_per_second=576.498Mi/s items_per_second=9.22397k/s
// LongDoubleStdSort/16384        527372 ns       527049 ns         1321 bytes_per_second=474.339Mi/s items_per_second=1.89736k/s
// LongDoubleStdSort/65536       2526744 ns      2525241 ns          282 bytes_per_second=396.002Mi/s items_per_second=396.002/s
// LongDoubleStdSort/262144     12533595 ns     12525652 ns           56 bytes_per_second=319.345Mi/s items_per_second=79.8362/s
// LongDoubleStdSort/1048576    58681460 ns     58648095 ns           11 bytes_per_second=272.814Mi/s items_per_second=17.0509/s
// LongDoubleStdSort/4194304   303218242 ns    303050780 ns            3 bytes_per_second=211.186Mi/s items_per_second=3.29978/s
// LongDoubleStdSort/16777216 1562130147 ns   1561208614 ns            1 bytes_per_second=163.976Mi/s items_per_second=0.640529/s
// LongDoubleStdSort/67108864 4608944330 ns   4606113994 ns            1 bytes_per_second=222.313Mi/s items_per_second=0.217103/s
// IntStdSort
//  3.06 │ a0:┌─→test     %r13,%r13                                                                                                                                         ▒
//        │    │↓ je       be                                                                                                                                                ▒
//   3.86 │    │  mov      %rbx,%rdi                                                                                                                                         ▒
//   5.78 │    │  mov      %r15,%rsi                                                                                                                                         ▒
//   2.46 │    │  mov      %r12,%rdx                                                                                                                                         ▒
//   7.48 │    │→ call     void std::__introsort_loop<int*, long, __gnu_cxx::__ops::_Iter_less_iter>(int*, int*, long, __gnu_cxx::__ops::_Iter_less_iter)                    ◆
//   5.35 │    │  mov      %rbx,%rdi                                                                                                                                         ▒
//   4.72 │    │  mov      %r15,%rsi                                                                                                                                         ▒
//  10.27 │    │→ call     void std::__final_insertion_sort<int*, __gnu_cxx::__ops::_Iter_less_iter>(int*, int*, __gnu_cxx::__ops::_Iter_less_iter)                          ▒
//   4.75 │ be:│  test     %r14,%r14                                                                                                                                         ▒
//   1.13 │    │↓ jle      227                                                                                                                                               ▒
//   2.79 │    │  dec      %r14                                                                                                                                              ▒
//   3.32 │    └──jne      a0 

// Note1: With test only 1 time each, clang++ perform better in small and medium size of array and worse at larger size of array.
BENCHMARK_MAIN();