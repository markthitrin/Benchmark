#include <benchmark/benchmark.h>

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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
    for(int q = 0;q < array_size;q++) {
      c[q] = a[q];
    }
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
// g++
// ---------------------------------------------------------------------------------------
// Benchmark                             Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------------
// CharAssignment/1                  0.724 ns        0.724 ns    961719894 bytes_per_second=1.28664Gi/s items_per_second=1.38152G/s
// CharAssignment/4                   1.69 ns         1.69 ns    414411616 bytes_per_second=2.2076Gi/s items_per_second=592.597M/s
// CharAssignment/16                  5.60 ns         5.60 ns    122982520 bytes_per_second=2.66008Gi/s items_per_second=178.515M/s
// CharAssignment/64                  22.6 ns         22.6 ns     30844570 bytes_per_second=2.64286Gi/s items_per_second=44.3399M/s
// CharAssignment/256                 94.0 ns         94.0 ns      7372114 bytes_per_second=2.53735Gi/s items_per_second=10.6424M/s
// CharAssignment/1024                 369 ns          369 ns      1912124 bytes_per_second=2.5839Gi/s items_per_second=2.70941M/s
// CharAssignment/4096                1456 ns         1456 ns       480699 bytes_per_second=2.6197Gi/s items_per_second=686.739k/s
// CharAssignment/16384               5819 ns         5818 ns       120113 bytes_per_second=2.62254Gi/s items_per_second=171.871k/s
// CharAssignment/65536              23362 ns        23360 ns        29971 bytes_per_second=2.61284Gi/s items_per_second=42.8088k/s
// CharAssignment/262144             93685 ns        93670 ns         7463 bytes_per_second=2.60639Gi/s items_per_second=10.6758k/s
// CharAssignment/1048576           375273 ns       375245 ns         1541 bytes_per_second=2.60246Gi/s items_per_second=2.66492k/s
//  15.74 │210:┌─→movzbl   (%rbx,%rax,1),%ecx                                                                                                                                   ▒
//  15.70 │    │  mov      -0x68(%rbp),%rdx                                                                                                                                     ▒
//  15.10 │    │  mov      %cl,(%rdx,%rax,1)                                                                                                                                    ▒
//  14.62 │    │  mov      %rax,%rdx                                                                                                                                            ▒
//  15.03 │    │  add      $0x1,%rax                                                                                                                                            ▒
//  15.35 │    ├──cmp      %rdx,%rsi                                                                                                                                            ◆
//        │    └──jne      210          

// ShortAssignment/1                  3.11 ns         3.11 ns    224712316 bytes_per_second=612.812Mi/s items_per_second=321.29M/s
// ShortAssignment/4                  2.63 ns         2.63 ns    265642369 bytes_per_second=2.829Gi/s items_per_second=379.701M/s
// ShortAssignment/16                 2.17 ns         2.17 ns    322999443 bytes_per_second=13.7512Gi/s items_per_second=461.414M/s
// ShortAssignment/64                 2.65 ns         2.65 ns    264238904 bytes_per_second=45.0074Gi/s items_per_second=377.55M/s
// ShortAssignment/256                5.57 ns         5.57 ns    125467535 bytes_per_second=85.5699Gi/s items_per_second=179.453M/s
// ShortAssignment/1024               16.0 ns         16.0 ns     43711299 bytes_per_second=119.09Gi/s items_per_second=62.4373M/s
// ShortAssignment/4096               62.5 ns         62.5 ns     11115387 bytes_per_second=122.021Gi/s items_per_second=15.9935M/s
// ShortAssignment/16384               517 ns          517 ns      1341683 bytes_per_second=59.0403Gi/s items_per_second=1.93463M/s
// ShortAssignment/65536              2060 ns         2059 ns       339325 bytes_per_second=59.2732Gi/s items_per_second=485.566k/s
// ShortAssignment/262144            10396 ns        10395 ns        65997 bytes_per_second=46.9746Gi/s items_per_second=96.2039k/s
// ShortAssignment/1048576          154561 ns       154543 ns         4466 bytes_per_second=12.6381Gi/s items_per_second=6.47071k/s
//   5.70 │ e0:   test     %r14d,%r14d
//        │     ↓ je       250
//   5.54 │ e9:   mov      -0x78(%rbp),%rdx
//   6.84 │       mov      -0x68(%rbp),%rdi
//   5.93 │       mov      %r12,%rsi
//  12.08 │     → call     memcpy@plt
//  21.61 │ f9:   test     %r15,%r15
//   0.16 │     ↓ jle      28d
//  20.15 │102:   sub      $0x1,%r15
//  20.38 │     ↑ jne      e0

// IntAssignment/1                    2.87 ns         2.87 ns    243576818 bytes_per_second=1.29601Gi/s items_per_second=347.895M/s
// IntAssignment/4                    2.41 ns         2.41 ns    290807764 bytes_per_second=6.18983Gi/s items_per_second=415.392M/s
// IntAssignment/16                   2.41 ns         2.41 ns    290617734 bytes_per_second=24.756Gi/s items_per_second=415.337M/s
// IntAssignment/64                   4.11 ns         4.11 ns    287736986 bytes_per_second=58.0332Gi/s items_per_second=243.409M/s
// IntAssignment/256                  8.34 ns         8.34 ns     73616397 bytes_per_second=114.316Gi/s items_per_second=119.869M/s
// IntAssignment/1024                 31.6 ns         31.6 ns     22161883 bytes_per_second=120.772Gi/s items_per_second=31.6596M/s
// IntAssignment/4096                  145 ns          145 ns      4797168 bytes_per_second=104.999Gi/s items_per_second=6.88121M/s
// IntAssignment/16384                1031 ns         1030 ns       677014 bytes_per_second=59.2328Gi/s items_per_second=970.47k/s
// IntAssignment/65536                4566 ns         4565 ns       153019 bytes_per_second=53.4768Gi/s items_per_second=219.041k/s
// IntAssignment/262144              21281 ns        21280 ns        32901 bytes_per_second=45.8902Gi/s items_per_second=46.9915k/s
// IntAssignment/1048576            372222 ns       372130 ns         1892 bytes_per_second=10.497Gi/s items_per_second=2.68723k/s

// LongLongAssignment/1               2.41 ns         2.41 ns    290727795 bytes_per_second=3.09498Gi/s items_per_second=415.401M/s
// LongLongAssignment/4               2.41 ns         2.41 ns    290655369 bytes_per_second=12.3739Gi/s items_per_second=415.198M/s
// LongLongAssignment/16              2.42 ns         2.42 ns    289971151 bytes_per_second=49.3569Gi/s items_per_second=414.036M/s
// LongLongAssignment/64              5.57 ns         5.57 ns    124993065 bytes_per_second=85.574Gi/s items_per_second=179.462M/s
// LongLongAssignment/256             16.0 ns         16.0 ns     43673837 bytes_per_second=119.065Gi/s items_per_second=62.4242M/s
// LongLongAssignment/1024            62.5 ns         62.5 ns     11110676 bytes_per_second=122.046Gi/s items_per_second=15.9968M/s
// LongLongAssignment/4096             517 ns          517 ns      1342044 bytes_per_second=58.99Gi/s items_per_second=1.93298M/s
// LongLongAssignment/16384           2115 ns         2115 ns       328671 bytes_per_second=57.7175Gi/s items_per_second=472.822k/s
// LongLongAssignment/65536          10611 ns        10610 ns        64177 bytes_per_second=46.0209Gi/s items_per_second=94.2508k/s
// LongLongAssignment/262144        139479 ns       139465 ns         5032 bytes_per_second=14.0044Gi/s items_per_second=7.17026k/s
// LongLongAssignment/1048576      1020216 ns      1020023 ns          652 bytes_per_second=7.65914Gi/s items_per_second=980.37/s

// FloatAssignment/1                  3.17 ns         3.17 ns    228286694 bytes_per_second=1.17409Gi/s items_per_second=315.168M/s
// FloatAssignment/4                  2.41 ns         2.41 ns    290691524 bytes_per_second=6.18872Gi/s items_per_second=415.318M/s
// FloatAssignment/16                 1.94 ns         1.94 ns    361863370 bytes_per_second=30.8004Gi/s items_per_second=516.745M/s
// FloatAssignment/64                 4.09 ns         4.09 ns    287547768 bytes_per_second=58.2319Gi/s items_per_second=244.242M/s
// FloatAssignment/256                9.48 ns         9.48 ns     84523818 bytes_per_second=100.614Gi/s items_per_second=105.501M/s
// FloatAssignment/1024               31.6 ns         31.6 ns     22115537 bytes_per_second=120.551Gi/s items_per_second=31.6017M/s
// FloatAssignment/4096                145 ns          145 ns      4805456 bytes_per_second=104.985Gi/s items_per_second=6.88027M/s
// FloatAssignment/16384              1032 ns         1031 ns       675795 bytes_per_second=59.172Gi/s items_per_second=969.474k/s
// FloatAssignment/65536              4477 ns         4477 ns       156914 bytes_per_second=54.5372Gi/s items_per_second=223.384k/s
// FloatAssignment/262144            22048 ns        22047 ns        32922 bytes_per_second=44.2954Gi/s items_per_second=45.3585k/s
// FloatAssignment/1048576          362286 ns       362245 ns         1916 bytes_per_second=10.7835Gi/s items_per_second=2.76056k/s

// DoubleAssignment/1                 2.65 ns         2.65 ns    264302902 bytes_per_second=2.81337Gi/s items_per_second=377.604M/s
// DoubleAssignment/4                 2.65 ns         2.65 ns    264216060 bytes_per_second=11.2515Gi/s items_per_second=377.537M/s
// DoubleAssignment/16                2.66 ns         2.66 ns    263358755 bytes_per_second=44.8594Gi/s items_per_second=376.308M/s
// DoubleAssignment/64                5.59 ns         5.59 ns    124925667 bytes_per_second=85.2995Gi/s items_per_second=178.886M/s
// DoubleAssignment/256               16.0 ns         16.0 ns     43602172 bytes_per_second=118.866Gi/s items_per_second=62.3198M/s
// DoubleAssignment/1024              62.7 ns         62.7 ns     11161813 bytes_per_second=121.77Gi/s items_per_second=15.9606M/s
// DoubleAssignment/4096               518 ns          518 ns      1345772 bytes_per_second=58.9569Gi/s items_per_second=1.9319M/s
// DoubleAssignment/16384             2060 ns         2060 ns       339957 bytes_per_second=59.2581Gi/s items_per_second=485.442k/s
// DoubleAssignment/65536            10386 ns        10385 ns        67426 bytes_per_second=47.018Gi/s items_per_second=96.2929k/s
// DoubleAssignment/262144          416562 ns       416500 ns         1580 bytes_per_second=4.68938Gi/s items_per_second=2.40096k/s
// DoubleAssignment/1048576        1632497 ns      1632038 ns          413 bytes_per_second=4.78696Gi/s items_per_second=612.731/s

// LongDoubleAssignment/1             2.81 ns         2.81 ns    262759856 bytes_per_second=5.29543Gi/s items_per_second=355.37M/s
// LongDoubleAssignment/4             2.18 ns         2.18 ns    321110698 bytes_per_second=27.3317Gi/s items_per_second=458.549M/s
// LongDoubleAssignment/16            4.16 ns         4.16 ns    261795271 bytes_per_second=57.3577Gi/s items_per_second=240.576M/s
// LongDoubleAssignment/64            8.43 ns         8.43 ns     72692682 bytes_per_second=113.098Gi/s items_per_second=118.592M/s
// LongDoubleAssignment/256           31.9 ns         31.9 ns     21601815 bytes_per_second=119.47Gi/s items_per_second=31.3183M/s
// LongDoubleAssignment/1024           146 ns          146 ns      4772328 bytes_per_second=104.527Gi/s items_per_second=6.85026M/s
// LongDoubleAssignment/4096          1038 ns         1037 ns       672967 bytes_per_second=58.8372Gi/s items_per_second=963.988k/s
// LongDoubleAssignment/16384         4661 ns         4660 ns       151977 bytes_per_second=52.3943Gi/s items_per_second=214.607k/s
// LongDoubleAssignment/65536        21464 ns        21458 ns        32320 bytes_per_second=45.5094Gi/s items_per_second=46.6016k/s
// LongDoubleAssignment/262144      635780 ns       635605 ns         1949 bytes_per_second=6.14572Gi/s items_per_second=1.5733k/s
// LongDoubleAssignment/1048576    3934847 ns      3934221 ns          299 bytes_per_second=3.97156Gi/s items_per_second=254.18/s




// clang++
// ---------------------------------------------------------------------------------------
// Benchmark                             Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------------
// CharAssignment/1                   1.21 ns         1.21 ns    578620517 bytes_per_second=790.077Mi/s items_per_second=828.456M/s
// CharAssignment/4                   1.94 ns         1.94 ns    361447776 bytes_per_second=1.92456Gi/s items_per_second=516.62M/s
// CharAssignment/16                  5.78 ns         5.78 ns    121039191 bytes_per_second=2.57926Gi/s items_per_second=173.091M/s
// CharAssignment/64                  22.9 ns         22.9 ns     30346206 bytes_per_second=2.60478Gi/s items_per_second=43.701M/s
// CharAssignment/256                 95.3 ns         95.3 ns      7293520 bytes_per_second=2.50235Gi/s items_per_second=10.4956M/s
// CharAssignment/1024                 373 ns          373 ns      1874483 bytes_per_second=2.55618Gi/s items_per_second=2.68035M/s
// CharAssignment/4096                1485 ns         1485 ns       471093 bytes_per_second=2.56843Gi/s items_per_second=673.299k/s
// CharAssignment/16384               5940 ns         5939 ns       117629 bytes_per_second=2.5694Gi/s items_per_second=168.388k/s
// CharAssignment/65536              23790 ns        23785 ns        29447 bytes_per_second=2.56611Gi/s items_per_second=42.0431k/s
// CharAssignment/262144            212676 ns       212637 ns         3300 bytes_per_second=1.14816Gi/s items_per_second=4.70286k/s
// CharAssignment/1048576           631879 ns       631789 ns         1302 bytes_per_second=1.54571Gi/s items_per_second=1.58281k/s
// 0.97 │ b0:   test     %r12d,%r12d                                                                                                                                          ▒
//      │     ↓ jle      125                                                                                                                                                  ▒
// 0.92 │       xor      %edi,%edi                                                                                                                                            ▒
// 0.87 │       cmp      $0x3,%rcx                                                                                                                                            ◆
//      │     ↓ jb       103                                                                                                                                                  ▒
// 0.41 │       nop                                                                                                                                                           ▒
// 4.66 │ c0:   movzbl   (%rbx,%rdi,1),%r8d                                                                                                                                   ▒
// 5.04 │       mov      -0x30(%rbp),%r9                                                                                                                                      ▒
// 9.07 │       mov      %r8b,(%r9,%rdi,1)                                                                                                                                    ▒
// 4.97 │       movzbl   0x1(%rbx,%rdi,1),%r8d                                                                                                                                ▒
// 4.82 │       mov      -0x30(%rbp),%r9                                                                                                                                      ▒
// 9.02 │       mov      %r8b,0x1(%r9,%rdi,1)                                                                                                                                 ▒ // 9.27 │       movzbl   0x2(%rbx,%rdi,1),%r8d                                                                                                                                ▒
// 4.92 │       mov      -0x30(%rbp),%r9                                                                                                                                      ▒
// 4.74 │       mov      %r8b,0x2(%r9,%rdi,1)
// 4.98 │       movzbl   0x3(%rbx,%rdi,1),%r8d                                                                                                                                ▒
// 4.75 │       mov      -0x30(%rbp),%r9                                                                                                                                      ▒
// 9.32 │       mov      %r8b,0x3(%r9,%rdi,1)                                                                                                                                 ▒
// 5.28 │       add      $0x4,%rdi                                                                                                                                            ▒
// 5.03 │       cmp      %rdi,%rax          

// ShortAssignment/1                  1.21 ns         1.21 ns    577204171 bytes_per_second=1.5376Gi/s items_per_second=825.492M/s
// ShortAssignment/4                  1.45 ns         1.45 ns    480266940 bytes_per_second=5.12338Gi/s items_per_second=687.648M/s
// ShortAssignment/16                 1.21 ns         1.21 ns    577185195 bytes_per_second=24.5906Gi/s items_per_second=825.122M/s
// ShortAssignment/64                 2.19 ns         2.19 ns    320458737 bytes_per_second=54.5158Gi/s items_per_second=457.311M/s
// ShortAssignment/256                8.02 ns         8.02 ns     86939089 bytes_per_second=59.4367Gi/s items_per_second=124.648M/s
// ShortAssignment/1024               31.3 ns         31.3 ns     22350115 bytes_per_second=60.889Gi/s items_per_second=31.9234M/s
// ShortAssignment/4096                127 ns          127 ns      5485142 bytes_per_second=60.0559Gi/s items_per_second=7.87164M/s
// ShortAssignment/16384               752 ns          752 ns       923970 bytes_per_second=40.5778Gi/s items_per_second=1.32965M/s
// ShortAssignment/65536              2990 ns         2990 ns       231892 bytes_per_second=40.8277Gi/s items_per_second=334.46k/s
// ShortAssignment/262144            12991 ns        12988 ns        53938 bytes_per_second=37.594Gi/s items_per_second=76.9925k/s
// ShortAssignment/1048576          164298 ns       164247 ns         4064 bytes_per_second=11.8914Gi/s items_per_second=6.0884k/s
//  10.60 │ f0:┌─→movupd   (%rbx,%r8,2),%xmm0                                                                                                                                   ▒
//  10.88 │    │  movups   0x10(%rbx,%r8,2),%xmm1                                                                                                                               ▒
//  12.15 │    │  movupd   %xmm0,(%rdi,%r8,2)                                                                                                                                   ▒
//  12.12 │    │  movups   %xmm1,0x10(%rdi,%r8,2)                                                                                                                               ▒
//  11.76 │    │  add      $0x10,%r8                                                                                                                                            ▒
//  12.06 │    ├──cmp      %r8,%rdx                                                                                                                                             ▒
//        │    └──jne      f0       

// IntAssignment/1                    1.45 ns         1.45 ns    481189373 bytes_per_second=2.57546Gi/s items_per_second=691.345M/s
// IntAssignment/4                    1.69 ns         1.69 ns    413729319 bytes_per_second=8.81064Gi/s items_per_second=591.272M/s
// IntAssignment/16                   2.04 ns         2.04 ns    342989815 bytes_per_second=29.1966Gi/s items_per_second=489.838M/s
// IntAssignment/64                   5.11 ns         5.10 ns    136106145 bytes_per_second=46.7096Gi/s items_per_second=195.914M/s
// IntAssignment/256                  18.5 ns         18.5 ns     35898597 bytes_per_second=51.5861Gi/s items_per_second=54.0919M/s
// IntAssignment/1024                 67.9 ns         67.9 ns     10281432 bytes_per_second=56.2163Gi/s items_per_second=14.7368M/s
// IntAssignment/4096                  259 ns          259 ns      2702356 bytes_per_second=58.9172Gi/s items_per_second=3.8612M/s
// IntAssignment/16384                1502 ns         1502 ns       466820 bytes_per_second=40.6421Gi/s items_per_second=665.88k/s
// IntAssignment/65536                6457 ns         6456 ns       108277 bytes_per_second=37.8181Gi/s items_per_second=154.903k/s
// IntAssignment/262144              31221 ns        31211 ns        20179 bytes_per_second=31.2886Gi/s items_per_second=32.0395k/s
// IntAssignment/1048576           1798672 ns      1798277 ns          337 bytes_per_second=2.17222Gi/s items_per_second=556.088/s

// LongLongAssignment/1               1.47 ns         1.47 ns    477896136 bytes_per_second=5.05922Gi/s items_per_second=679.037M/s
// LongLongAssignment/4               1.48 ns         1.48 ns    468853688 bytes_per_second=20.1911Gi/s items_per_second=677.5M/s
// LongLongAssignment/16              3.10 ns         3.09 ns    224389248 bytes_per_second=38.5187Gi/s items_per_second=323.119M/s
// LongLongAssignment/64              10.2 ns         10.2 ns     69470890 bytes_per_second=46.5765Gi/s items_per_second=97.678M/s
// LongLongAssignment/256             34.8 ns         34.8 ns     21276527 bytes_per_second=54.8112Gi/s items_per_second=28.7369M/s
// LongLongAssignment/1024             133 ns          133 ns      5313032 bytes_per_second=57.3739Gi/s items_per_second=7.52011M/s
// LongLongAssignment/4096             764 ns          764 ns       909741 bytes_per_second=39.9618Gi/s items_per_second=1.30947M/s
// LongLongAssignment/16384           3061 ns         3060 ns       231283 bytes_per_second=39.8868Gi/s items_per_second=326.753k/s
// LongLongAssignment/65536          13250 ns        13246 ns        52619 bytes_per_second=36.8616Gi/s items_per_second=75.4926k/s
// LongLongAssignment/262144        279597 ns       279575 ns         4235 bytes_per_second=6.98605Gi/s items_per_second=3.57686k/s
// LongLongAssignment/1048576      3386696 ns      3385538 ns          332 bytes_per_second=2.30761Gi/s items_per_second=295.374/s

// FloatAssignment/1                  1.23 ns         1.23 ns    569337462 bytes_per_second=3.03796Gi/s items_per_second=815.495M/s
// FloatAssignment/4                  1.47 ns         1.47 ns    475283219 bytes_per_second=10.1198Gi/s items_per_second=679.131M/s
// FloatAssignment/16                 2.11 ns         2.11 ns    318602160 bytes_per_second=28.3116Gi/s items_per_second=474.99M/s
// FloatAssignment/64                 5.13 ns         5.13 ns    135641944 bytes_per_second=46.495Gi/s items_per_second=195.014M/s
// FloatAssignment/256                17.0 ns         17.0 ns     41441103 bytes_per_second=56.1048Gi/s items_per_second=58.8302M/s
// FloatAssignment/1024               68.7 ns         68.7 ns     10118455 bytes_per_second=55.4987Gi/s items_per_second=14.5486M/s
// FloatAssignment/4096                264 ns          264 ns      2666161 bytes_per_second=57.8735Gi/s items_per_second=3.79279M/s
// FloatAssignment/16384              1514 ns         1513 ns       462472 bytes_per_second=40.3281Gi/s items_per_second=660.735k/s
// FloatAssignment/65536              6451 ns         6451 ns       106396 bytes_per_second=37.8466Gi/s items_per_second=155.02k/s
// FloatAssignment/262144            28984 ns        28980 ns        25402 bytes_per_second=33.698Gi/s items_per_second=34.5067k/s
// FloatAssignment/1048576         1573923 ns      1573510 ns          613 bytes_per_second=2.48251Gi/s items_per_second=635.522/s

// DoubleAssignment/1                 1.47 ns         1.47 ns    474608863 bytes_per_second=5.07716Gi/s items_per_second=681.445M/s
// DoubleAssignment/4                 1.23 ns         1.23 ns    569147461 bytes_per_second=24.2987Gi/s items_per_second=815.33M/s
// DoubleAssignment/16                2.21 ns         2.21 ns    317358029 bytes_per_second=53.9201Gi/s items_per_second=452.314M/s
// DoubleAssignment/64                8.13 ns         8.13 ns     86335115 bytes_per_second=58.6458Gi/s items_per_second=122.989M/s
// DoubleAssignment/256               32.0 ns         32.0 ns     22237913 bytes_per_second=59.5673Gi/s items_per_second=31.2304M/s
// DoubleAssignment/1024               128 ns          128 ns      5430876 bytes_per_second=59.444Gi/s items_per_second=7.79144M/s
// DoubleAssignment/4096               758 ns          758 ns       919069 bytes_per_second=40.2747Gi/s items_per_second=1.31972M/s
// DoubleAssignment/16384             3411 ns         3410 ns       211868 bytes_per_second=35.7941Gi/s items_per_second=293.225k/s
// DoubleAssignment/65536            13344 ns        13342 ns        49741 bytes_per_second=36.596Gi/s items_per_second=74.9485k/s
// DoubleAssignment/262144          341436 ns       341319 ns         2961 bytes_per_second=5.72229Gi/s items_per_second=2.92981k/s
// DoubleAssignment/1048576        3440485 ns      3439248 ns          325 bytes_per_second=2.27157Gi/s items_per_second=290.761/s

// LongDoubleAssignment/1             2.93 ns         2.93 ns    239618910 bytes_per_second=5.09183Gi/s items_per_second=341.707M/s
// LongDoubleAssignment/4             9.22 ns         9.22 ns     75378773 bytes_per_second=6.46488Gi/s items_per_second=108.463M/s
// LongDoubleAssignment/16            35.6 ns         35.6 ns     19840596 bytes_per_second=6.70503Gi/s items_per_second=28.123M/s
// LongDoubleAssignment/64             140 ns          140 ns      4969619 bytes_per_second=6.80008Gi/s items_per_second=7.1304M/s
// LongDoubleAssignment/256            560 ns          560 ns      1239632 bytes_per_second=6.81519Gi/s items_per_second=1.78656M/s
// LongDoubleAssignment/1024          2249 ns         2249 ns       312711 bytes_per_second=6.78573Gi/s items_per_second=444.71k/s
// LongDoubleAssignment/4096          8965 ns         8963 ns        77689 bytes_per_second=6.80959Gi/s items_per_second=111.568k/s
// LongDoubleAssignment/16384        36152 ns        36145 ns        19532 bytes_per_second=6.75455Gi/s items_per_second=27.6666k/s
// LongDoubleAssignment/65536       145417 ns       145379 ns         4814 bytes_per_second=6.71734Gi/s items_per_second=6.87856k/s
// LongDoubleAssignment/262144     1573969 ns      1573224 ns          556 bytes_per_second=2.48296Gi/s items_per_second=635.637/s
// LongDoubleAssignment/1048576    6311568 ns      6310685 ns          123 bytes_per_second=2.47596Gi/s items_per_second=158.461/s
//   6.51 │100:┌─→fldt     -0x30(%rbx,%r10,1)                                                                                                                                   ▒
//  10.24 │    │  fstpt    -0x30(%r8,%r10,1)                                                                                                                                    ▒
//   8.71 │    │  fldt     -0x20(%rbx,%r10,1)                                                                                                                                   ▒
//  19.00 │    │  fstpt    -0x20(%r8,%r10,1)                                                                                                                                    ◆
//  11.60 │    │  fldt     -0x10(%rbx,%r10,1)                                                                                                                                   ▒
//  10.98 │    │  fstpt    -0x10(%r8,%r10,1)                                                                                                                                    ▒
//   8.65 │    │  fldt     (%rbx,%r10,1)                                                                                                                                        ▒
//  10.83 │    │  fstpt    (%r8,%r10,1)                                                                                                                                         ▒
//   0.65 │    │  add      $0x4,%r9                                                                                                                                             ▒
//   0.73 │    │  add      $0x40,%r10                                                                                                                                           ▒
//   0.67 │    ├──cmp      %r9,%rax                                                                                                                                             ▒
//        │    └──jne      100     


// Note1: g++ char array copy doesn't is the only one that isn't implement the memcpy call
// Note2: clang++ char                - direct copy with unroll loop
//                short,int,float,... - use %xmm implementation which get better performance thatn g+= in some cases.
//                LongDouble          - implement some weird thing that is quite slow.
// Note3: memcpy will be better than normally assignment when the space is big enough but get slower when thet reach some limmit of memmory hit.
//        which is still faster thatn normal approach.

BENCHMARK_MAIN();