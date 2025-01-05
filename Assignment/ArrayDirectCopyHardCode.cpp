#define BENCHMARK_TEMPLATE_RANGE(func) \
    BENCHMARK(func<1>);\
    BENCHMARK(func<4>);\
    BENCHMARK(func<16>);\
    BENCHMARK(func<64>);\
    BENCHMARK(func<256>);\
    BENCHMARK(func<1024>);
#define EXPRESSION1(index) c[index] = a[index];
#define EXPRESSION2(index) EXPRESSION1(index) EXPRESSION1(index + 1)
#define EXPRESSION4(index) EXPRESSION2(index) EXPRESSION2(index + 2)
#define EXPRESSION8(index) EXPRESSION4(index) EXPRESSION4(index + 4)
#define EXPRESSION16(index) EXPRESSION8(index) EXPRESSION8(index + 8)
#define EXPRESSION32(index) EXPRESSION16(index) EXPRESSION16(index + 16)
#define EXPRESSION64(index) EXPRESSION32(index) EXPRESSION32(index + 32)
#define EXPRESSION128(index) EXPRESSION64(index) EXPRESSION64(index + 64)
#define EXPRESSION256(index) EXPRESSION128(index) EXPRESSION128(index + 128)
#define EXPRESSION512(index) EXPRESSION256(index) EXPRESSION256(index + 256)
#define EXPRESSION1024(index) EXPRESSION512(index) EXPRESSION512(index + 512)
#define EXPRESSION(array_size) \
    if constexpr(array_size == 1) {EXPRESSION1(0)}\
    else if constexpr(array_size == 4) {EXPRESSION4(0)}\
    else if constexpr(array_size == 16) {EXPRESSION16(0)}\
    else if constexpr(array_size == 64) {EXPRESSION64(0)}\
    else if constexpr(array_size == 256) {EXPRESSION256(0)}\
    else if constexpr(array_size == 1024) {EXPRESSION1024(0)}\

#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

template<int array_size>
static void CharAssignment(benchmark::State& state) {
  char a[array_size];
  char c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

template<int array_size>
static void ShortAssignment(benchmark::State& state) {
  short a[array_size];
  short c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

template<int array_size>
static void IntAssignment(benchmark::State& state) {
  int a[array_size];
  int c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

template<int array_size>
static void LongLongAssignment(benchmark::State& state) {
  long long a[array_size];
  long long c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

template<int array_size>
static void FloatAssignment(benchmark::State& state) {
  float a[array_size];
  float c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

template<int array_size>
static void DoubleAssignment(benchmark::State& state) {
  double a[array_size];
  double c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
}

template<int array_size>
static void LongDoubleAssignment(benchmark::State& state) {
  long double a[array_size];
  long double c[array_size];
  for(int q = 0 ;q < array_size;q++) {
    a[q] = std::rand();
  }
  for(auto _ : state) {
    EXPRESSION(array_size)
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
}

BENCHMARK_TEMPLATE_RANGE(CharAssignment);
BENCHMARK_TEMPLATE_RANGE(ShortAssignment);
BENCHMARK_TEMPLATE_RANGE(IntAssignment);
BENCHMARK_TEMPLATE_RANGE(LongLongAssignment);
BENCHMARK_TEMPLATE_RANGE(FloatAssignment);
BENCHMARK_TEMPLATE_RANGE(DoubleAssignment);
BENCHMARK_TEMPLATE_RANGE(LongDoubleAssignment);

// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAssignment<1>               0.239 ns        0.238 ns   2913928266 bytes_per_second=3.90691Gi/s items_per_second=4.19501G/s
// CharAssignment<4>               0.240 ns        0.240 ns   2917184844 bytes_per_second=15.5317Gi/s items_per_second=4.16927G/s
// CharAssignment<16>              0.241 ns        0.241 ns   2902607163 bytes_per_second=61.792Gi/s items_per_second=4.14679G/s
// CharAssignment<64>              0.966 ns        0.965 ns    724924627 bytes_per_second=61.7498Gi/s items_per_second=1.03599G/s
// CharAssignment<256>              3.86 ns         3.85 ns    181389131 bytes_per_second=61.8525Gi/s items_per_second=259.428M/s
// CharAssignment<1024>             15.4 ns         15.4 ns     45411083 bytes_per_second=61.9013Gi/s items_per_second=64.9083M/s

// ShortAssignment<1>              0.239 ns        0.239 ns   2918693955 bytes_per_second=7.78954Gi/s items_per_second=4.18198G/s
// ShortAssignment<4>              0.240 ns        0.240 ns   2919813861 bytes_per_second=31.0759Gi/s items_per_second=4.17094G/s
// ShortAssignment<16>             0.485 ns        0.484 ns   1444480155 bytes_per_second=61.514Gi/s items_per_second=2.06407G/s
// ShortAssignment<64>              1.93 ns         1.93 ns    362502535 bytes_per_second=61.7316Gi/s items_per_second=517.842M/s
// ShortAssignment<256>             7.73 ns         7.72 ns     90472609 bytes_per_second=61.7755Gi/s items_per_second=129.553M/s
// ShortAssignment<1024>            30.9 ns         30.8 ns     22696181 bytes_per_second=61.8693Gi/s items_per_second=32.4373M/s

// IntAssignment<1>                0.480 ns        0.480 ns   1457890526 bytes_per_second=7.76122Gi/s items_per_second=2.08339G/s
// IntAssignment<4>                0.240 ns        0.240 ns   2918683465 bytes_per_second=62.0946Gi/s items_per_second=4.1671G/s
// IntAssignment<16>               0.969 ns        0.968 ns    722541637 bytes_per_second=61.5521Gi/s items_per_second=1.03267G/s
// IntAssignment<64>                3.86 ns         3.86 ns    181290406 bytes_per_second=61.7448Gi/s items_per_second=258.977M/s
// IntAssignment<256>               15.5 ns         15.4 ns     45330650 bytes_per_second=61.7653Gi/s items_per_second=64.7656M/s
// IntAssignment<1024>              61.7 ns         61.7 ns     11312888 bytes_per_second=61.8732Gi/s items_per_second=16.2197M/s

// LongLongAssignment<1>           0.240 ns        0.239 ns   2916101526 bytes_per_second=31.1284Gi/s items_per_second=4.17798G/s
// LongLongAssignment<4>           0.482 ns        0.482 ns   1452138348 bytes_per_second=61.8179Gi/s items_per_second=2.07426G/s
// LongLongAssignment<16>           1.93 ns         1.93 ns    362868103 bytes_per_second=61.7585Gi/s items_per_second=518.068M/s
// LongLongAssignment<64>           7.72 ns         7.71 ns     90508459 bytes_per_second=61.8382Gi/s items_per_second=129.684M/s
// LongLongAssignment<256>          30.8 ns         30.8 ns     22723128 bytes_per_second=61.9315Gi/s items_per_second=32.4699M/s
// LongLongAssignment<1024>          123 ns          123 ns      5670020 bytes_per_second=61.8546Gi/s items_per_second=8.10741M/s

// FloatAssignment<1>              0.240 ns        0.240 ns   2905848170 bytes_per_second=15.5441Gi/s items_per_second=4.17259G/s
// FloatAssignment<4>              0.240 ns        0.240 ns   2918911723 bytes_per_second=62.159Gi/s items_per_second=4.17142G/s
// FloatAssignment<16>             0.969 ns        0.969 ns    722808258 bytes_per_second=61.5369Gi/s items_per_second=1.03242G/s
// FloatAssignment<64>              3.86 ns         3.86 ns    181413668 bytes_per_second=61.7484Gi/s items_per_second=258.992M/s
// FloatAssignment<256>             15.4 ns         15.4 ns     45316045 bytes_per_second=61.811Gi/s items_per_second=64.8136M/s
// FloatAssignment<1024>            61.7 ns         61.7 ns     11322426 bytes_per_second=61.8539Gi/s items_per_second=16.2146M/s

// DoubleAssignment<1>             0.240 ns        0.240 ns   2918564232 bytes_per_second=31.0224Gi/s items_per_second=4.16376G/s
// DoubleAssignment<4>             0.483 ns        0.483 ns   1449604008 bytes_per_second=61.7147Gi/s items_per_second=2.0708G/s
// DoubleAssignment<16>             1.93 ns         1.93 ns    362375014 bytes_per_second=61.7134Gi/s items_per_second=517.69M/s
// DoubleAssignment<64>             7.73 ns         7.72 ns     90438345 bytes_per_second=61.7385Gi/s items_per_second=129.475M/s
// DoubleAssignment<256>            30.9 ns         30.9 ns     22674907 bytes_per_second=61.8201Gi/s items_per_second=32.4116M/s
// DoubleAssignment<1024>            124 ns          123 ns      5657591 bytes_per_second=61.786Gi/s items_per_second=8.09841M/s

// LongDoubleAssignment<1>          1.21 ns         1.21 ns    577836705 bytes_per_second=12.3252Gi/s items_per_second=827.128M/s
// LongDoubleAssignment<4>          8.68 ns         8.67 ns     80425133 bytes_per_second=6.87089Gi/s items_per_second=115.274M/s
// LongDoubleAssignment<16>         34.7 ns         34.7 ns     20143286 bytes_per_second=6.87404Gi/s items_per_second=28.8318M/s
// LongDoubleAssignment<64>          141 ns          140 ns      5034727 bytes_per_second=6.78924Gi/s items_per_second=7.11903M/s
// LongDoubleAssignment<256>         559 ns          558 ns      1245387 bytes_per_second=6.83053Gi/s items_per_second=1.79058M/s
// LongDoubleAssignment<1024>       2235 ns         2234 ns       312905 bytes_per_second=6.83115Gi/s items_per_second=447.686k/s

// CharAssignment<16>
//  16.69 │190:   sub      $0x1,%rax
//  16.69 │     ↑ je       55
//  16.71 │19a:   movdqa   -0x60(%rbp),%xmm0
//  33.25 │       movaps   %xmm0,-0x50(%rbp)
//  16.66 │       test     %rax,%rax


// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAssignment<1>                1.18 ns         1.18 ns    590135396 bytes_per_second=805.746Mi/s items_per_second=844.886M/s
// CharAssignment<4>                1.89 ns         1.89 ns    369633628 bytes_per_second=1.9673Gi/s items_per_second=528.092M/s
// CharAssignment<16>               4.03 ns         4.03 ns    173695252 bytes_per_second=3.69637Gi/s items_per_second=248.059M/s
// CharAssignment<64>               1.19 ns         1.19 ns    587038072 bytes_per_second=50.096Gi/s items_per_second=840.472M/s
// CharAssignment<256>              4.03 ns         4.03 ns    173717960 bytes_per_second=59.228Gi/s items_per_second=248.42M/s
// CharAssignment<1024>             15.5 ns         15.5 ns     45110562 bytes_per_second=61.6403Gi/s items_per_second=64.6346M/s

// ShortAssignment<1>               1.18 ns         1.18 ns    591249434 bytes_per_second=1.57373Gi/s items_per_second=844.888M/s
// ShortAssignment<4>               1.19 ns         1.19 ns    589518711 bytes_per_second=6.2795Gi/s items_per_second=842.82M/s
// ShortAssignment<16>              4.03 ns         4.03 ns    173821355 bytes_per_second=7.40378Gi/s items_per_second=248.43M/s
// ShortAssignment<64>              2.13 ns         2.13 ns    328106915 bytes_per_second=55.8942Gi/s items_per_second=468.874M/s
// ShortAssignment<256>             7.85 ns         7.85 ns     88930403 bytes_per_second=60.7371Gi/s items_per_second=127.375M/s
// ShortAssignment<1024>            30.7 ns         30.7 ns     22798431 bytes_per_second=62.1174Gi/s items_per_second=32.5674M/s

// IntAssignment<1>                0.476 ns        0.476 ns   1469824452 bytes_per_second=7.82339Gi/s items_per_second=2.10008G/s
// IntAssignment<4>                 1.19 ns         1.19 ns    589588553 bytes_per_second=12.5713Gi/s items_per_second=843.649M/s
// IntAssignment<16>                4.03 ns         4.03 ns    173883341 bytes_per_second=14.8017Gi/s items_per_second=248.332M/s
// IntAssignment<64>                4.03 ns         4.03 ns    173761483 bytes_per_second=59.2323Gi/s items_per_second=248.438M/s
// IntAssignment<256>               15.5 ns         15.5 ns     45224450 bytes_per_second=61.6436Gi/s items_per_second=64.638M/s
// IntAssignment<1024>              61.2 ns         61.2 ns     11406435 bytes_per_second=62.3678Gi/s items_per_second=16.3493M/s

// LongLongAssignment<1>           0.476 ns        0.476 ns   1470286681 bytes_per_second=15.6499Gi/s items_per_second=2.1005G/s
// LongLongAssignment<4>            1.19 ns         1.19 ns    588219911 bytes_per_second=25.0747Gi/s items_per_second=841.368M/s
// LongLongAssignment<16>           4.05 ns         4.04 ns    173348756 bytes_per_second=29.4759Gi/s items_per_second=247.262M/s
// LongLongAssignment<64>           7.87 ns         7.87 ns     88749640 bytes_per_second=60.5824Gi/s items_per_second=127.05M/s
// LongLongAssignment<256>          30.8 ns         30.8 ns     22747836 bytes_per_second=61.9955Gi/s items_per_second=32.5035M/s
// LongLongAssignment<1024>          122 ns          122 ns      5713839 bytes_per_second=62.343Gi/s items_per_second=8.17142M/s

// FloatAssignment<1>              0.477 ns        0.477 ns   1467152183 bytes_per_second=7.80843Gi/s items_per_second=2.09606G/s
// FloatAssignment<4>              0.477 ns        0.477 ns   1465995969 bytes_per_second=31.2295Gi/s items_per_second=2.09577G/s
// FloatAssignment<16>              1.19 ns         1.19 ns    586353435 bytes_per_second=49.9804Gi/s items_per_second=838.532M/s
// FloatAssignment<64>              4.04 ns         4.04 ns    173452685 bytes_per_second=59.0686Gi/s items_per_second=247.752M/s
// FloatAssignment<256>             15.5 ns         15.5 ns     45114285 bytes_per_second=61.5038Gi/s items_per_second=64.4914M/s
// FloatAssignment<1024>            61.3 ns         61.3 ns     11385411 bytes_per_second=62.2252Gi/s items_per_second=16.312M/s

// DoubleAssignment<1>             0.477 ns        0.477 ns   1467393359 bytes_per_second=15.6061Gi/s items_per_second=2.09462G/s
// DoubleAssignment<4>              1.19 ns         1.19 ns    586442760 bytes_per_second=24.9887Gi/s items_per_second=838.481M/s
// DoubleAssignment<16>             4.04 ns         4.04 ns    173440208 bytes_per_second=29.5434Gi/s items_per_second=247.828M/s
// DoubleAssignment<64>             7.87 ns         7.87 ns     88737772 bytes_per_second=60.5811Gi/s items_per_second=127.048M/s
// DoubleAssignment<256>            30.7 ns         30.7 ns     22746867 bytes_per_second=62.0386Gi/s items_per_second=32.5261M/s
// DoubleAssignment<1024>            122 ns          122 ns      5687420 bytes_per_second=62.3966Gi/s items_per_second=8.17845M/s

// LongDoubleAssignment<1>          1.44 ns         1.44 ns    486487379 bytes_per_second=10.3639Gi/s items_per_second=695.507M/s
// LongDoubleAssignment<4>          5.04 ns         5.04 ns    138610554 bytes_per_second=11.8333Gi/s items_per_second=198.53M/s
// LongDoubleAssignment<16>         28.7 ns         28.7 ns     24502330 bytes_per_second=8.32145Gi/s items_per_second=34.9027M/s
// LongDoubleAssignment<64>          138 ns          138 ns      5080135 bytes_per_second=6.93409Gi/s items_per_second=7.27092M/s
// LongDoubleAssignment<256>         551 ns          551 ns      1266828 bytes_per_second=6.92622Gi/s items_per_second=1.81567M/s
// LongDoubleAssignment<1024>       2206 ns         2205 ns       316918 bytes_per_second=6.91951Gi/s items_per_second=453.477k/s
// IntAssignment<4>
// 10.53 │ 60:┌─→mov      %r14d,-0x70(%rbp)                                                                                                                                 ▒
//  19.92 │    │  mov      %r15d,-0x6c(%rbp)                                                                                                                                 ▒
//  16.49 │    │  mov      %r12d,-0x68(%rbp)                                                                                                                                 ▒
//  10.80 │    │  mov      %r13d,-0x64(%rbp)                                                                                                                                 ▒
//  13.45 │    │  mov      %rax,-0x80(%rbp)                                                                                                                                  ▒
//   8.39 │    │  test     %rbx,%rbx                                                                                                                                         ▒
//        │    │↓ jle      1d8                                                                                                                                               ▒
//  10.68 │    │  dec      %rbx                                                                                                                                              ▒
//   9.75 │    └──jne      60     
// ShortAssignment<16>
  // 2.56 │120:┌─→mov      %cx,-0x80(%rbp)                                                                                                                                   ▒
  // 5.62 │    │  mov      %dx,-0x7e(%rbp)                                                                                                                                   ▒
  // 6.34 │    │  mov      %si,-0x7c(%rbp)                                                                                                                                   ▒
  // 3.08 │    │  mov      %di,-0x7a(%rbp)                                                                                                                                   ▒
  // 4.26 │    │  mov      %r8w,-0x78(%rbp)                                                                                                                                  ▒
  // 7.06 │    │  mov      %r9w,-0x76(%rbp)                                                                                                                                  ▒
  // 8.81 │    │  mov      %r10w,-0x74(%rbp)                                                                                                                                 ▒
  // 3.28 │    │  mov      %r11w,-0x72(%rbp)                                                                                                                                 ▒
  // 1.90 │    │  mov      -0x30(%rbp),%eax                                                                                                                                  ▒
  // 3.26 │    │  mov      %ax,-0x70(%rbp)                                                                                                                                   ▒
  // 4.31 │    │  mov      -0xb0(%rbp),%eax                                                                                                                                  ▒
  // 1.79 │    │  mov      %ax,-0x6e(%rbp)                                                                                                                                   ▒
  // 2.56 │    │  mov      -0xac(%rbp),%eax                                                                                                                                  ▒
  // 3.78 │    │  mov      %ax,-0x6c(%rbp)                                                                                                                                   ▒
  // 4.20 │    │  mov      -0xa8(%rbp),%eax                                                                                                                                  ◆
  // 1.64 │    │  mov      %ax,-0x6a(%rbp)                                                                                                                                   ▒
  // 4.09 │    │  mov      %r12w,-0x68(%rbp)                                                                                                                                 ▒
  // 7.12 │    │  mov      %r13w,-0x66(%rbp)                                                                                                                                 ▒
  // 7.65 │    │  mov      %bx,-0x64(%rbp)                                                                                                                                   ▒
  // 3.69 │    │  mov      %r14w,-0x62(%rbp)                                                                                                                                 ▒
  // 2.64 │    │  lea      -0x80(%rbp),%rax                                                                                                                                  ▒
  // 2.30 │    │  mov      %rax,-0xc0(%rbp)                                                                                                                                  ▒
  // 2.34 │    │  test     %r15,%r15                                                                                                                                         ▒
  //      │    │↓ jle      2f3                                                                                                                                               ▒
  // 3.59 │    │  dec      %r15                                                                                                                                              ▒
  // 2.14 │    └──jne      120    
// CharAssignment<64>
// 10.00 │ 80:┌─→movapd   %xmm0,-0x90(%rbp)                                                                                                                                 ▒
//  19.51 │    │  movaps   %xmm1,-0x80(%rbp)                                                                                                                                 ▒
//  17.00 │    │  movaps   %xmm2,-0x70(%rbp)                                                                                                                                 ▒
//  10.52 │    │  movaps   %xmm3,-0x60(%rbp)                                                                                                                                 ▒
//  13.21 │    │  mov      %rax,-0x50(%rbp)                                                                                                                                  ▒
//   8.76 │    │  test     %r14,%r14                                                                                                                                         ▒
//        │    │↓ jle      1fa                                                                                                                                               ▒
//  10.99 │    │  dec      %r14                                                                                                                                              ▒
//  10.00 │    └──jne      80 

// Note1: The g++ version is faster in some small case sine the clang++ use regular mov instead of mov256 bit at the same time.

BENCHMARK_MAIN();