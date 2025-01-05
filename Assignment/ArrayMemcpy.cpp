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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
  delete[] a;
  delete[] c;
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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
  delete[] a;
  delete[] c;
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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
  delete[] a;
  delete[] c;
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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
  delete[] a;
  delete[] c;
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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
  delete[] a;
  delete[] c;
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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
  delete[] a;
  delete[] c;
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
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
  delete[] a;
  delete[] c;
}

BENCHMARK(CharAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(ShortAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(IntAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(LongLongAssignment)->RangeMultiplier(4)->Range(1, 1<<20); 
BENCHMARK(FloatAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(DoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(LongDoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
//g++
// ---------------------------------------------------------------------------------------
// Benchmark                             Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------------
// CharAssignment/1                   3.12 ns         3.12 ns    224178279 bytes_per_second=305.885Mi/s items_per_second=320.743M/s
// CharAssignment/4                   2.88 ns         2.88 ns    242747096 bytes_per_second=1.29407Gi/s items_per_second=347.374M/s
// CharAssignment/16                  2.42 ns         2.42 ns    290887121 bytes_per_second=6.15957Gi/s items_per_second=413.362M/s
// CharAssignment/64                  1.95 ns         1.95 ns    360489973 bytes_per_second=30.5964Gi/s items_per_second=513.323M/s
// CharAssignment/256                 4.12 ns         4.12 ns    286165428 bytes_per_second=57.9203Gi/s items_per_second=242.936M/s
// CharAssignment/1024                8.32 ns         8.32 ns     72440780 bytes_per_second=114.661Gi/s items_per_second=120.23M/s
// CharAssignment/4096                31.8 ns         31.8 ns     22023806 bytes_per_second=120.051Gi/s items_per_second=31.4708M/s
// CharAssignment/16384                150 ns          150 ns      4659164 bytes_per_second=101.63Gi/s items_per_second=6.66043M/s
// CharAssignment/65536               1041 ns         1040 ns       672222 bytes_per_second=58.6718Gi/s items_per_second=961.279k/s
// CharAssignment/262144              4585 ns         4583 ns       153119 bytes_per_second=53.2658Gi/s items_per_second=218.177k/s
// CharAssignment/1048576            21339 ns        21334 ns        32830 bytes_per_second=45.7751Gi/s items_per_second=46.8737k/s
//   8.39 │ b8:   mov      -0x68(%rbp),%rdi
//   8.74 │       mov      %r13,%rdx
//   8.54 │       mov      %r12,%rsi
//  28.08 │     → call     memcpy@plt
//  22.17 │ c7:   sub      $0x1,%r14
//  21.82 │     ↑ jne      b8

// ShortAssignment/1                  2.42 ns         2.42 ns    288943967 bytes_per_second=787.657Mi/s items_per_second=412.959M/s
// ShortAssignment/4                  2.41 ns         2.41 ns    289895539 bytes_per_second=3.0861Gi/s items_per_second=414.21M/s
// ShortAssignment/16                 1.94 ns         1.94 ns    360450622 bytes_per_second=15.3539Gi/s items_per_second=515.191M/s
// ShortAssignment/64                 2.43 ns         2.43 ns    287891817 bytes_per_second=49.0888Gi/s items_per_second=411.786M/s
// ShortAssignment/256                5.62 ns         5.61 ns    124384006 bytes_per_second=84.9343Gi/s items_per_second=178.12M/s
// ShortAssignment/1024               16.1 ns         16.1 ns     43386470 bytes_per_second=118.214Gi/s items_per_second=61.9783M/s
// ShortAssignment/4096               63.0 ns         63.0 ns     11059540 bytes_per_second=121.09Gi/s items_per_second=15.8715M/s
// ShortAssignment/16384               522 ns          522 ns      1333718 bytes_per_second=58.4982Gi/s items_per_second=1.91687M/s
// ShortAssignment/65536              2082 ns         2082 ns       337222 bytes_per_second=58.6342Gi/s items_per_second=480.331k/s
// ShortAssignment/262144            10378 ns        10376 ns        67108 bytes_per_second=47.0566Gi/s items_per_second=96.372k/s
// ShortAssignment/1048576          167077 ns       167016 ns         4269 bytes_per_second=11.6942Gi/s items_per_second=5.98745k/s

// IntAssignment/1                    3.69 ns         3.69 ns    181285742 bytes_per_second=1.01045Gi/s items_per_second=271.239M/s
// IntAssignment/4                    2.42 ns         2.41 ns    289757220 bytes_per_second=6.17068Gi/s items_per_second=414.107M/s
// IntAssignment/16                   1.94 ns         1.94 ns    360097438 bytes_per_second=30.6791Gi/s items_per_second=514.71M/s
// IntAssignment/64                   4.12 ns         4.12 ns    286627840 bytes_per_second=57.9061Gi/s items_per_second=242.876M/s
// IntAssignment/256                  9.52 ns         9.52 ns     83582564 bytes_per_second=100.17Gi/s items_per_second=105.035M/s
// IntAssignment/1024                 31.8 ns         31.8 ns     22042040 bytes_per_second=120.119Gi/s items_per_second=31.4884M/s
// IntAssignment/4096                  150 ns          150 ns      4658921 bytes_per_second=101.736Gi/s items_per_second=6.66736M/s
// IntAssignment/16384                1039 ns         1039 ns       672508 bytes_per_second=58.7592Gi/s items_per_second=962.711k/s
// IntAssignment/65536                4777 ns         4776 ns       146131 bytes_per_second=51.118Gi/s items_per_second=209.379k/s
// IntAssignment/262144              21156 ns        21150 ns        32549 bytes_per_second=46.1728Gi/s items_per_second=47.281k/s
// IntAssignment/1048576            352381 ns       352269 ns         2045 bytes_per_second=11.0888Gi/s items_per_second=2.83874k/s

// LongLongAssignment/1               2.42 ns         2.42 ns    289246511 bytes_per_second=3.07759Gi/s items_per_second=413.067M/s
// LongLongAssignment/4               2.15 ns         2.15 ns    289153709 bytes_per_second=13.8368Gi/s items_per_second=464.285M/s
// LongLongAssignment/16              2.43 ns         2.43 ns    288242980 bytes_per_second=49.0625Gi/s items_per_second=411.566M/s
// LongLongAssignment/64              5.62 ns         5.62 ns    124440007 bytes_per_second=84.8488Gi/s items_per_second=177.941M/s
// LongLongAssignment/256             16.1 ns         16.1 ns     43465686 bytes_per_second=118.367Gi/s items_per_second=62.0583M/s
// LongLongAssignment/1024            63.0 ns         63.0 ns     11067442 bytes_per_second=121.042Gi/s items_per_second=15.8652M/s
// LongLongAssignment/4096             521 ns          521 ns      1339027 bytes_per_second=58.5487Gi/s items_per_second=1.91852M/s
// LongLongAssignment/16384           2066 ns         2066 ns       339555 bytes_per_second=59.0939Gi/s items_per_second=484.098k/s
// LongLongAssignment/65536          10255 ns        10252 ns        66500 bytes_per_second=47.6257Gi/s items_per_second=97.5373k/s
// LongLongAssignment/262144        160980 ns       160934 ns         4356 bytes_per_second=12.1362Gi/s items_per_second=6.21374k/s
// LongLongAssignment/1048576      1091918 ns      1091633 ns          580 bytes_per_second=7.15671Gi/s items_per_second=916.059/s

// FloatAssignment/1                  2.58 ns         2.58 ns    275308348 bytes_per_second=1.44467Gi/s items_per_second=387.801M/s
// FloatAssignment/4                  2.42 ns         2.42 ns    289331140 bytes_per_second=6.16008Gi/s items_per_second=413.396M/s
// FloatAssignment/16                 1.94 ns         1.94 ns    360097179 bytes_per_second=30.6747Gi/s items_per_second=514.637M/s
// FloatAssignment/64                 4.12 ns         4.12 ns    286893432 bytes_per_second=57.8817Gi/s items_per_second=242.774M/s
// FloatAssignment/256                8.32 ns         8.31 ns     73416759 bytes_per_second=114.702Gi/s items_per_second=120.274M/s
// FloatAssignment/1024               31.8 ns         31.8 ns     22008395 bytes_per_second=120.11Gi/s items_per_second=31.486M/s
// FloatAssignment/4096                150 ns          150 ns      4662058 bytes_per_second=101.805Gi/s items_per_second=6.6719M/s
// FloatAssignment/16384              1039 ns         1039 ns       672798 bytes_per_second=58.7351Gi/s items_per_second=962.316k/s
// FloatAssignment/65536              4726 ns         4726 ns       153333 bytes_per_second=51.6624Gi/s items_per_second=211.609k/s
// FloatAssignment/262144            21503 ns        21498 ns        31926 bytes_per_second=45.425Gi/s items_per_second=46.5152k/s
// FloatAssignment/1048576          404106 ns       403974 ns         1728 bytes_per_second=9.66955Gi/s items_per_second=2.4754k/s

// DoubleAssignment/1                 2.42 ns         2.42 ns    289102946 bytes_per_second=3.07566Gi/s items_per_second=412.809M/s
// DoubleAssignment/4                 2.42 ns         2.42 ns    288849033 bytes_per_second=12.3106Gi/s items_per_second=413.074M/s
// DoubleAssignment/16                2.43 ns         2.43 ns    287964346 bytes_per_second=49.0903Gi/s items_per_second=411.799M/s
// DoubleAssignment/64                5.62 ns         5.62 ns    124495281 bytes_per_second=84.8979Gi/s items_per_second=178.044M/s
// DoubleAssignment/256               16.1 ns         16.1 ns     43383049 bytes_per_second=118.334Gi/s items_per_second=62.041M/s
// DoubleAssignment/1024              64.4 ns         64.4 ns     10782020 bytes_per_second=118.518Gi/s items_per_second=15.5344M/s
// DoubleAssignment/4096               522 ns          521 ns      1337594 bytes_per_second=58.5279Gi/s items_per_second=1.91784M/s
// DoubleAssignment/16384             2162 ns         2162 ns       324063 bytes_per_second=56.4732Gi/s items_per_second=462.629k/s
// DoubleAssignment/65536            10442 ns        10439 ns        66616 bytes_per_second=46.7753Gi/s items_per_second=95.7959k/s
// DoubleAssignment/262144          179058 ns       178998 ns         4173 bytes_per_second=10.9115Gi/s items_per_second=5.58667k/s
// DoubleAssignment/1048576        1202969 ns      1202430 ns          570 bytes_per_second=6.49726Gi/s items_per_second=831.649/s

// LongDoubleAssignment/1             2.42 ns         2.42 ns    289275071 bytes_per_second=6.15778Gi/s items_per_second=413.242M/s
// LongDoubleAssignment/4             2.43 ns         2.43 ns    288742704 bytes_per_second=24.5774Gi/s items_per_second=412.34M/s
// LongDoubleAssignment/16            4.12 ns         4.12 ns    286495238 bytes_per_second=57.914Gi/s items_per_second=242.909M/s
// LongDoubleAssignment/64            9.52 ns         9.52 ns     84112158 bytes_per_second=100.157Gi/s items_per_second=105.022M/s
// LongDoubleAssignment/256           31.7 ns         31.7 ns     22059321 bytes_per_second=120.215Gi/s items_per_second=31.5135M/s
// LongDoubleAssignment/1024           150 ns          149 ns      4669574 bytes_per_second=102.08Gi/s items_per_second=6.68988M/s
// LongDoubleAssignment/4096          1040 ns         1040 ns       669989 bytes_per_second=58.6781Gi/s items_per_second=961.382k/s
// LongDoubleAssignment/16384         4586 ns         4585 ns       150891 bytes_per_second=53.244Gi/s items_per_second=218.087k/s
// LongDoubleAssignment/65536        21502 ns        21498 ns        33103 bytes_per_second=45.4252Gi/s items_per_second=46.5154k/s
// LongDoubleAssignment/262144      336707 ns       336607 ns         2053 bytes_per_second=11.6048Gi/s items_per_second=2.97083k/s
// LongDoubleAssignment/1048576    2428202 ns      2427497 ns          285 bytes_per_second=6.43667Gi/s items_per_second=411.947/s


// clang++
// ---------------------------------------------------------------------------------------
// Benchmark                             Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------------
// CharAssignment/1                   2.66 ns         2.66 ns    243547020 bytes_per_second=359.02Mi/s items_per_second=376.459M/s
// CharAssignment/4                   2.64 ns         2.64 ns    257587232 bytes_per_second=1.40983Gi/s items_per_second=378.447M/s
// CharAssignment/16                  2.42 ns         2.42 ns    288920491 bytes_per_second=6.15038Gi/s items_per_second=412.745M/s
// CharAssignment/64                  1.95 ns         1.95 ns    359103395 bytes_per_second=30.5735Gi/s items_per_second=512.938M/s
// CharAssignment/256                 4.36 ns         4.36 ns    285409777 bytes_per_second=54.6621Gi/s items_per_second=229.27M/s
// CharAssignment/1024                8.59 ns         8.59 ns     71366818 bytes_per_second=110.996Gi/s items_per_second=116.387M/s
// CharAssignment/4096                32.1 ns         32.1 ns     21788765 bytes_per_second=118.737Gi/s items_per_second=31.1262M/s
// CharAssignment/16384                151 ns          151 ns      4640922 bytes_per_second=101.195Gi/s items_per_second=6.63195M/s
// CharAssignment/65536               1045 ns         1045 ns       670020 bytes_per_second=58.4197Gi/s items_per_second=957.148k/s
// CharAssignment/262144              4797 ns         4796 ns       145110 bytes_per_second=50.9092Gi/s items_per_second=208.524k/s
// CharAssignment/1048576            21562 ns        21556 ns        34530 bytes_per_second=45.3027Gi/s items_per_second=46.39k/s
//   8.00 │ 80:   mov      -0x58(%rbp),%rdi
//   7.92 │       mov      %rbx,%rsi
//   9.70 │       mov      %r14,%rdx
//  24.96 │     → call     memcpy@plt
//   9.79 │       mov      %r13,-0x68(%rbp)
//  13.69 │       test     %r12,%r12
//        │     ↓ jle      209
//  13.31 │       dec      %r12
//  10.31 │     ↑ jne      80

// ShortAssignment/1                  2.91 ns         2.91 ns    240845466 bytes_per_second=656.172Mi/s items_per_second=344.023M/s
// ShortAssignment/4                  2.85 ns         2.85 ns    207667429 bytes_per_second=2.61125Gi/s items_per_second=350.477M/s
// ShortAssignment/16                 2.43 ns         2.43 ns    288924481 bytes_per_second=12.2883Gi/s items_per_second=412.327M/s
// ShortAssignment/64                 2.44 ns         2.44 ns    287066282 bytes_per_second=48.883Gi/s items_per_second=410.061M/s
// ShortAssignment/256                5.97 ns         5.96 ns    111952394 bytes_per_second=79.9445Gi/s items_per_second=167.656M/s
// ShortAssignment/1024               16.8 ns         16.8 ns     42589809 bytes_per_second=113.62Gi/s items_per_second=59.5694M/s
// ShortAssignment/4096               63.5 ns         63.5 ns     10988227 bytes_per_second=120.111Gi/s items_per_second=15.7432M/s
// ShortAssignment/16384               524 ns          524 ns      1331710 bytes_per_second=58.2113Gi/s items_per_second=1.90747M/s
// ShortAssignment/65536              2085 ns         2085 ns       335628 bytes_per_second=58.5556Gi/s items_per_second=479.688k/s
// ShortAssignment/262144            10633 ns        10630 ns        66453 bytes_per_second=45.9331Gi/s items_per_second=94.071k/s
// ShortAssignment/1048576          158846 ns       158791 ns         4407 bytes_per_second=12.3Gi/s items_per_second=6.29758k/s

// IntAssignment/1                    3.01 ns         3.01 ns    238022854 bytes_per_second=1.23795Gi/s items_per_second=332.31M/s
// IntAssignment/4                    2.42 ns         2.42 ns    288884930 bytes_per_second=6.15038Gi/s items_per_second=412.745M/s
// IntAssignment/16                   2.43 ns         2.43 ns    287893999 bytes_per_second=24.5315Gi/s items_per_second=411.571M/s
// IntAssignment/64                   4.37 ns         4.37 ns    284728353 bytes_per_second=54.6085Gi/s items_per_second=229.045M/s
// IntAssignment/256                  9.81 ns         9.81 ns     81380905 bytes_per_second=97.2127Gi/s items_per_second=101.935M/s
// IntAssignment/1024                 32.2 ns         32.1 ns     21777755 bytes_per_second=118.663Gi/s items_per_second=31.1067M/s
// IntAssignment/4096                  152 ns          152 ns      4617103 bytes_per_second=100.71Gi/s items_per_second=6.60012M/s
// IntAssignment/16384                1045 ns         1044 ns       670332 bytes_per_second=58.4443Gi/s items_per_second=957.551k/s
// IntAssignment/65536                4700 ns         4699 ns       148574 bytes_per_second=51.9581Gi/s items_per_second=212.82k/s
// IntAssignment/262144              21283 ns        21278 ns        32562 bytes_per_second=45.8959Gi/s items_per_second=46.9974k/s
// IntAssignment/1048576            364179 ns       364039 ns         1840 bytes_per_second=10.7303Gi/s items_per_second=2.74695k/s

// LongLongAssignment/1               2.42 ns         2.42 ns    288948737 bytes_per_second=3.07495Gi/s items_per_second=412.713M/s
// LongLongAssignment/4               1.95 ns         1.95 ns    358997440 bytes_per_second=15.2582Gi/s items_per_second=511.98M/s
// LongLongAssignment/16              2.44 ns         2.44 ns    286934595 bytes_per_second=48.8876Gi/s items_per_second=410.099M/s
// LongLongAssignment/64              5.89 ns         5.88 ns    118953029 bytes_per_second=81.0408Gi/s items_per_second=169.955M/s
// LongLongAssignment/256             16.4 ns         16.4 ns     42586806 bytes_per_second=116.096Gi/s items_per_second=60.868M/s
// LongLongAssignment/1024            63.5 ns         63.5 ns     11009650 bytes_per_second=120.113Gi/s items_per_second=15.7435M/s
// LongLongAssignment/4096             525 ns          525 ns      1329255 bytes_per_second=58.0948Gi/s items_per_second=1.90365M/s
// LongLongAssignment/16384           2155 ns         2154 ns       324903 bytes_per_second=56.6717Gi/s items_per_second=464.255k/s
// LongLongAssignment/65536          10515 ns        10512 ns        66413 bytes_per_second=46.4485Gi/s items_per_second=95.1265k/s
// LongLongAssignment/262144        157441 ns       157403 ns         4504 bytes_per_second=12.4085Gi/s items_per_second=6.35314k/s
// LongLongAssignment/1048576      1218887 ns      1218513 ns          564 bytes_per_second=6.4115Gi/s items_per_second=820.672/s

// FloatAssignment/1                  2.91 ns         2.91 ns    240835394 bytes_per_second=1.28118Gi/s items_per_second=343.914M/s
// FloatAssignment/4                  2.42 ns         2.42 ns    288877312 bytes_per_second=6.14973Gi/s items_per_second=412.701M/s
// FloatAssignment/16                 1.95 ns         1.95 ns    358730226 bytes_per_second=30.5431Gi/s items_per_second=512.427M/s
// FloatAssignment/64                 4.37 ns         4.37 ns    284946204 bytes_per_second=54.5485Gi/s items_per_second=228.793M/s
// FloatAssignment/256                8.61 ns         8.61 ns     71315765 bytes_per_second=110.827Gi/s items_per_second=116.21M/s
// FloatAssignment/1024               32.2 ns         32.2 ns     21778122 bytes_per_second=118.63Gi/s items_per_second=31.0981M/s
// FloatAssignment/4096                151 ns          151 ns      4619895 bytes_per_second=101.14Gi/s items_per_second=6.62834M/s
// FloatAssignment/16384              1044 ns         1044 ns       669940 bytes_per_second=58.4881Gi/s items_per_second=958.27k/s
// FloatAssignment/65536              4756 ns         4755 ns       146197 bytes_per_second=51.3452Gi/s items_per_second=210.31k/s
// FloatAssignment/262144            21469 ns        21464 ns        32479 bytes_per_second=45.4983Gi/s items_per_second=46.5903k/s
// FloatAssignment/1048576          380963 ns       380865 ns         1838 bytes_per_second=10.2563Gi/s items_per_second=2.6256k/s

// DoubleAssignment/1                 2.42 ns         2.42 ns    285129270 bytes_per_second=3.07378Gi/s items_per_second=412.556M/s
// DoubleAssignment/4                 1.95 ns         1.95 ns    358862254 bytes_per_second=15.2729Gi/s items_per_second=512.475M/s
// DoubleAssignment/16                2.47 ns         2.47 ns    286569626 bytes_per_second=48.2213Gi/s items_per_second=404.509M/s
// DoubleAssignment/64                5.89 ns         5.88 ns    118849436 bytes_per_second=81.0334Gi/s items_per_second=169.939M/s
// DoubleAssignment/256               16.4 ns         16.4 ns     42566342 bytes_per_second=116.048Gi/s items_per_second=60.8426M/s
// DoubleAssignment/1024              63.5 ns         63.5 ns     10996651 bytes_per_second=120.106Gi/s items_per_second=15.7425M/s
// DoubleAssignment/4096               525 ns          525 ns      1328916 bytes_per_second=58.1482Gi/s items_per_second=1.9054M/s
// DoubleAssignment/16384             2093 ns         2093 ns       334508 bytes_per_second=58.3308Gi/s items_per_second=477.846k/s
// DoubleAssignment/65536            10502 ns        10500 ns        66516 bytes_per_second=46.5023Gi/s items_per_second=95.2367k/s
// DoubleAssignment/262144          161102 ns       161055 ns         4488 bytes_per_second=12.1271Gi/s items_per_second=6.20906k/s
// DoubleAssignment/1048576        1191821 ns      1191522 ns          571 bytes_per_second=6.55674Gi/s items_per_second=839.262/s

// LongDoubleAssignment/1             2.43 ns         2.42 ns    289001834 bytes_per_second=6.14527Gi/s items_per_second=412.402M/s
// LongDoubleAssignment/4             1.95 ns         1.95 ns    357898196 bytes_per_second=30.5212Gi/s items_per_second=512.061M/s
// LongDoubleAssignment/16            4.37 ns         4.37 ns    284829098 bytes_per_second=54.5209Gi/s items_per_second=228.677M/s
// LongDoubleAssignment/64            9.81 ns         9.81 ns     81434627 bytes_per_second=97.2007Gi/s items_per_second=101.922M/s
// LongDoubleAssignment/256           32.2 ns         32.1 ns     21775223 bytes_per_second=118.709Gi/s items_per_second=31.1188M/s
// LongDoubleAssignment/1024           151 ns          151 ns      4640665 bytes_per_second=101.09Gi/s items_per_second=6.62502M/s
// LongDoubleAssignment/4096          1046 ns         1046 ns       671561 bytes_per_second=58.3672Gi/s items_per_second=956.288k/s
// LongDoubleAssignment/16384         4744 ns         4743 ns       148449 bytes_per_second=51.4764Gi/s items_per_second=210.847k/s
// LongDoubleAssignment/65536        21384 ns        21378 ns        35329 bytes_per_second=45.6802Gi/s items_per_second=46.7766k/s
// LongDoubleAssignment/262144      372941 ns       372831 ns         1863 bytes_per_second=10.4773Gi/s items_per_second=2.68218k/s
// LongDoubleAssignment/1048576    2352472 ns      2351662 ns          291 bytes_per_second=6.64424Gi/s items_per_second=425.231/s


// both g++ and clang++ memcpy seem to be very consistent, with same speed and bandwidth behavior regardless the type.
BENCHMARK_MAIN();