#include <benchmark/benchmark.h>
#include <vector>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<char> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<short> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<int> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long long> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<float> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<double> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long double> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
}

BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);

// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   16.0 ns         16.0 ns     43509282 bytes_per_second=59.4604Mi/s items_per_second=62.3488M/s
// CharAllocate/4                   56.0 ns         56.0 ns     12213650 bytes_per_second=68.0817Mi/s items_per_second=17.8472M/s
// CharAllocate/16                   105 ns          105 ns      6640307 bytes_per_second=145.77Mi/s items_per_second=9.55317M/s
// CharAllocate/64                   193 ns          193 ns      3622599 bytes_per_second=316.102Mi/s items_per_second=5.17901M/s
// CharAllocate/256                  497 ns          497 ns      1409354 bytes_per_second=491.127Mi/s items_per_second=2.01165M/s
// CharAllocate/1024                1654 ns         1654 ns       417578 bytes_per_second=590.497Mi/s items_per_second=604.669k/s
// CharAllocate/4096                6233 ns         6232 ns       111836 bytes_per_second=626.799Mi/s items_per_second=160.461k/s
// CharAllocate/16384              32784 ns        32778 ns        28911 bytes_per_second=476.692Mi/s items_per_second=30.5083k/s
// CharAllocate/65536             150830 ns       150804 ns         4643 bytes_per_second=414.446Mi/s items_per_second=6.63113k/s
// CharAllocate/262144            775659 ns       775487 ns          898 bytes_per_second=322.378Mi/s items_per_second=1.28951k/s
// CharAllocate/1048576          3323098 ns      3322263 ns          210 bytes_per_second=301Mi/s items_per_second=301/s

// ShortAllocate/1                  16.0 ns         16.0 ns     44181244 bytes_per_second=119.301Mi/s items_per_second=62.5481M/s
// ShortAllocate/4                  58.2 ns         58.1 ns     11978393 bytes_per_second=131.214Mi/s items_per_second=17.1985M/s
// ShortAllocate/16                  107 ns          107 ns      6543292 bytes_per_second=285.23Mi/s items_per_second=9.34641M/s
// ShortAllocate/64                  196 ns          196 ns      3580576 bytes_per_second=623.963Mi/s items_per_second=5.11151M/s
// ShortAllocate/256                 506 ns          506 ns      1387578 bytes_per_second=965.027Mi/s items_per_second=1.97638M/s
// ShortAllocate/1024               1712 ns         1711 ns       411088 bytes_per_second=1.11458Gi/s items_per_second=584.359k/s
// ShortAllocate/4096               6325 ns         6323 ns       106404 bytes_per_second=1.20659Gi/s items_per_second=158.15k/s
// ShortAllocate/16384             24558 ns        24554 ns        28506 bytes_per_second=1.24288Gi/s items_per_second=40.7268k/s
// ShortAllocate/65536             97485 ns        97462 ns         7167 bytes_per_second=1.25249Gi/s items_per_second=10.2604k/s
// ShortAllocate/262144           505657 ns       505582 ns         1000 bytes_per_second=988.959Mi/s items_per_second=1.97792k/s
// ShortAllocate/1048576         3796988 ns      3795548 ns          166 bytes_per_second=526.933Mi/s items_per_second=263.467/s

// IntAllocate/1                    16.6 ns         16.6 ns     42162587 bytes_per_second=229.406Mi/s items_per_second=60.1373M/s
// IntAllocate/4                    61.7 ns         61.7 ns     11145620 bytes_per_second=247.369Mi/s items_per_second=16.2116M/s
// IntAllocate/16                    108 ns          108 ns      6459845 bytes_per_second=564.947Mi/s items_per_second=9.2561M/s
// IntAllocate/64                    177 ns          177 ns      3945994 bytes_per_second=1.34448Gi/s items_per_second=5.63915M/s
// IntAllocate/256                   383 ns          383 ns      1828164 bytes_per_second=2.49132Gi/s items_per_second=2.61234M/s
// IntAllocate/1024                 1074 ns         1073 ns       652415 bytes_per_second=3.55366Gi/s items_per_second=931.57k/s
// IntAllocate/4096                 3562 ns         3562 ns       196483 bytes_per_second=4.28425Gi/s items_per_second=280.772k/s
// IntAllocate/16384               13517 ns        13515 ns        51773 bytes_per_second=4.51612Gi/s items_per_second=73.9922k/s
// IntAllocate/65536               93120 ns        93099 ns         8922 bytes_per_second=2.62238Gi/s items_per_second=10.7413k/s
// IntAllocate/262144             342386 ns       342344 ns         1903 bytes_per_second=2.85258Gi/s items_per_second=2.92104k/s
// IntAllocate/1048576           5383122 ns      5381328 ns          105 bytes_per_second=743.311Mi/s items_per_second=185.828/s

// LongLongAllocate/1               17.6 ns         17.6 ns     35623140 bytes_per_second=433.605Mi/s items_per_second=56.8334M/s
// LongLongAllocate/4               55.6 ns         55.6 ns     12523268 bytes_per_second=549.028Mi/s items_per_second=17.9906M/s
// LongLongAllocate/16               102 ns          102 ns      6614912 bytes_per_second=1.16502Gi/s items_per_second=9.77294M/s
// LongLongAllocate/64               175 ns          175 ns      4002475 bytes_per_second=2.72798Gi/s items_per_second=5.72099M/s
// LongLongAllocate/256              410 ns          409 ns      1705939 bytes_per_second=4.6583Gi/s items_per_second=2.44229M/s
// LongLongAllocate/1024            1146 ns         1146 ns       611098 bytes_per_second=6.65671Gi/s items_per_second=872.508k/s
// LongLongAllocate/4096            3913 ns         3912 ns       179215 bytes_per_second=7.80017Gi/s items_per_second=255.596k/s
// LongLongAllocate/16384          14709 ns        14707 ns        47606 bytes_per_second=8.30009Gi/s items_per_second=67.9943k/s
// LongLongAllocate/65536          98539 ns        98520 ns         6771 bytes_per_second=4.95616Gi/s items_per_second=10.1502k/s
// LongLongAllocate/262144        483144 ns       482987 ns         1174 bytes_per_second=4.04385Gi/s items_per_second=2.07045k/s
// LongLongAllocate/1048576     12236450 ns     12234086 ns           55 bytes_per_second=653.911Mi/s items_per_second=81.7388/s

// FloatAllocate/1                  16.8 ns         16.8 ns     41604525 bytes_per_second=226.97Mi/s items_per_second=59.4988M/s
// FloatAllocate/4                  62.4 ns         62.4 ns     10927570 bytes_per_second=244.449Mi/s items_per_second=16.0202M/s
// FloatAllocate/16                  110 ns          110 ns      6262981 bytes_per_second=553.513Mi/s items_per_second=9.06875M/s
// FloatAllocate/64                  179 ns          179 ns      3886381 bytes_per_second=1.33008Gi/s items_per_second=5.57874M/s
// FloatAllocate/256                 366 ns          366 ns      1905825 bytes_per_second=2.60553Gi/s items_per_second=2.7321M/s
// FloatAllocate/1024               1074 ns         1074 ns       650632 bytes_per_second=3.55258Gi/s items_per_second=931.288k/s
// FloatAllocate/4096               3565 ns         3565 ns       196114 bytes_per_second=4.28076Gi/s items_per_second=280.544k/s
// FloatAllocate/16384             13516 ns        13513 ns        51754 bytes_per_second=4.51664Gi/s items_per_second=74.0006k/s
// FloatAllocate/65536             53289 ns        53280 ns        12928 bytes_per_second=4.58219Gi/s items_per_second=18.7687k/s
// FloatAllocate/262144           340673 ns       340612 ns         1876 bytes_per_second=2.86708Gi/s items_per_second=2.93589k/s
// FloatAllocate/1048576         2203313 ns      2202650 ns          306 bytes_per_second=1.77343Gi/s items_per_second=453.999/s

// DoubleAllocate/1                 17.8 ns         17.8 ns     38087189 bytes_per_second=428.283Mi/s items_per_second=56.1359M/s
// DoubleAllocate/4                 64.8 ns         64.8 ns     10633420 bytes_per_second=471.038Mi/s items_per_second=15.435M/s
// DoubleAllocate/16                 118 ns          118 ns      6026910 bytes_per_second=1.01299Gi/s items_per_second=8.49758M/s
// DoubleAllocate/64                 192 ns          192 ns      3651490 bytes_per_second=2.48818Gi/s items_per_second=5.2181M/s
// DoubleAllocate/256                467 ns          467 ns      1501694 bytes_per_second=4.0843Gi/s items_per_second=2.14135M/s
// DoubleAllocate/1024              1201 ns         1201 ns       581588 bytes_per_second=6.35231Gi/s items_per_second=832.609k/s
// DoubleAllocate/4096              3959 ns         3958 ns       176470 bytes_per_second=7.7099Gi/s items_per_second=252.638k/s
// DoubleAllocate/16384            19330 ns        19328 ns        47559 bytes_per_second=6.31587Gi/s items_per_second=51.7396k/s
// DoubleAllocate/65536            98367 ns        98349 ns         7132 bytes_per_second=4.96476Gi/s items_per_second=10.1678k/s
// DoubleAllocate/262144          530325 ns       530152 ns         1361 bytes_per_second=3.68408Gi/s items_per_second=1.88625k/s
// DoubleAllocate/1048576       11572134 ns     11567279 ns           55 bytes_per_second=691.606Mi/s items_per_second=86.4508/s

// LongDoubleAllocate/1             18.3 ns         18.3 ns     38266295 bytes_per_second=834.946Mi/s items_per_second=54.719M/s
// LongDoubleAllocate/4             59.5 ns         59.5 ns     11709622 bytes_per_second=1.00192Gi/s items_per_second=16.8094M/s
// LongDoubleAllocate/16             132 ns          132 ns      5293420 bytes_per_second=1.80409Gi/s items_per_second=7.56688M/s
// LongDoubleAllocate/64             331 ns          331 ns      2110555 bytes_per_second=2.87881Gi/s items_per_second=3.01865M/s
// LongDoubleAllocate/256           1130 ns         1129 ns       620145 bytes_per_second=3.37748Gi/s items_per_second=885.387k/s
// LongDoubleAllocate/1024          3988 ns         3987 ns       175504 bytes_per_second=3.82686Gi/s items_per_second=250.797k/s
// LongDoubleAllocate/4096         15381 ns        15377 ns        45459 bytes_per_second=3.96937Gi/s items_per_second=65.0341k/s
// LongDoubleAllocate/16384        61052 ns        61036 ns        11433 bytes_per_second=3.99996Gi/s items_per_second=16.3838k/s
// LongDoubleAllocate/65536       244984 ns       244927 ns         2867 bytes_per_second=3.98716Gi/s items_per_second=4.08285k/s
// LongDoubleAllocate/262144     1803380 ns      1802861 ns          389 bytes_per_second=2.1667Gi/s items_per_second=554.674/s
// LongDoubleAllocate/1048576   24268986 ns     24260301 ns           29 bytes_per_second=659.514Mi/s items_per_second=41.2196/s
// CharAllocate
//  6.87 │1b0:   movb     $0x0,(%rsi)                                                                                                                                       ▒
//   6.14 │       add      $0x1,%ebx                                                                                                                                         ▒
//   6.96 │       add      $0x1,%rsi                                                                                                                                         ▒
//  12.10 │       mov      %rsi,-0x78(%rbp)                                                                                                                                  ▒
//   6.32 │       cmp      %ebx,%r12d                                                                                                                                        ▒
//        │     ↓ je       1f0                                                                                                                                               ▒
//  10.37 │1c3:   mov      -0x78(%rbp),%rsi                                                                                                                                  ▒
//  18.57 │       mov      -0x70(%rbp),%rax                                                                                                                                  ◆
//  18.03 │1cb:   movb     $0x0,-0x81(%rbp)                                                                                                                                  ▒
//  10.42 │       cmp      %rax,%rsi 
//   0.01 │     ↑ jne      1b0                                                                                                                                               ▒
//   0.23 │       mov      -0x98(%rbp),%rdi                                                                                                                                  ▒
//   0.26 │       mov      %r15,%rdx                                                                                                                                         ▒
//   0.53 │     → call     void std::vector<char, std::allocator<char> >::_M_realloc_insert<char>(__gnu_cxx::__normal_iterator<char*, std::vector<char, std::allocator<char> ▒
//   0.60 │       add      $0x1,%ebx                                                                                                                                         ▒
//   0.53 │       cmp      %ebx,%r12d
// IntAllocate
//   7.31 │1b8:┌─→movl     $0x0,(%rsi)                                                                                                                                       ▒
//   7.47 │    │  add      $0x1,%ebx                                                                                                                                         ▒
//   8.96 │    │  add      $0x4,%rsi                                                                                                                                         ▒
//  13.33 │    │  mov      %rsi,-0x78(%rbp)                                                                                                                                  ▒
//   6.70 │    │  cmp      %ebx,%r12d                                                                                                                                        ▒
//   0.01 │    │↓ je       200                                                                                                                                               ▒
//   9.17 │1ce:│  mov      -0x78(%rbp),%rsi                                                                                                                                  ▒
//  11.19 │    │  mov      -0x70(%rbp),%rax                                                                                                                                  ▒
//  10.61 │1d6:│  movl     $0x0,-0x84(%rbp)                                                                                                                                  ◆
//  20.83 │    ├──cmp      %rax,%rsi                                                                                                                                         ▒
//        │    └──jne      1b8 
//              │    │↑ jne      1b8                                                                                                                                               ▒
//   0.25 │    │  mov      -0x98(%rbp),%rdi                                                                                                                                  ▒
//   0.37 │    │  mov      %r15,%rdx                                                                                                                                         ▒
//   0.80 │    │→ call     void std::vector<int, std::allocator<int> >::_M_realloc_insert<int>(__gnu_cxx::__normal_iterator<int*, std::vector<int, std::allocator<int> > >, i▒
//   0.24 │    │  add      $0x1,%ebx                                                                                                                                         ▒
//   0.30 │    ├──cmp      %ebx,%r12d                                                                                                                                        ▒
//        │    └──jne      1ce
// LongDoubleAllocate
//  26.56 │1a0:   fstpt    (%rsi)                                                                                                                                            ▒
//   1.16 │       add      $0x1,%r12d                                                                                                                                        ▒
//   1.14 │       add      $0x10,%rsi                                                                                                                                        ▒
//   1.11 │       mov      %rsi,-0x78(%rbp)                                                                                                                                  ▒
//   2.08 │       cmp      %r12d,%r13d                                                                                                                                       ▒
//        │     ↓ je       1f0                                                                                                                                               ▒
//   1.18 │1b3:   mov      -0x78(%rbp),%rsi                                                                                                                                  ▒
//   3.48 │       mov      -0x70(%rbp),%rax                                                                                                                                  ◆
//   3.20 │1bb:   fldz                                                                                                                                                       ▒
//  30.00 │       fstpt    -0x90(%rbp)                                                                                                                                       ▒
//  21.68 │       fldt     -0x90(%rbp)                                                                                                                                       ▒
//   6.33 │       cmp      %rax,%rsi
//    0.08 │    └──jne      1a0                                                                                                                                               ▒
//   0.12 │       fstp     %st(0)                                                                                                                                            ▒
//   0.09 │       lea      -0x90(%rbp),%rdx                                                                                                                                  ▒
//   0.09 │       lea      -0x80(%rbp),%rdi                                                                                                                                  ▒
//   0.19 │     → call     void std::vector<long double, std::allocator<long double> >::_M_realloc_insert<long double>(__gnu_cxx::__normal_iterator<long double*, std::vector▒
//   0.14 │       add      $0x1,%r12d                                                                                                                                        ▒
//   0.11 │       cmp      %r12d,%r13d                                                                                                                                       ▒
//        │     ↑ jne      1b3


// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   12.4 ns         12.4 ns     56448495 bytes_per_second=76.8546Mi/s items_per_second=80.5879M/s
// CharAllocate/4                   46.5 ns         46.5 ns     14904774 bytes_per_second=81.9928Mi/s items_per_second=21.4939M/s
// CharAllocate/16                  85.3 ns         85.3 ns      8538727 bytes_per_second=178.89Mi/s items_per_second=11.7237M/s
// CharAllocate/64                   128 ns          128 ns      5440034 bytes_per_second=475.955Mi/s items_per_second=7.79804M/s
// CharAllocate/256                  237 ns          237 ns      3001522 bytes_per_second=1.00646Gi/s items_per_second=4.22139M/s
// CharAllocate/1024                 534 ns          534 ns      1287801 bytes_per_second=1.78622Gi/s items_per_second=1.87298M/s
// CharAllocate/4096                1701 ns         1701 ns       411733 bytes_per_second=2.24323Gi/s items_per_second=588.049k/s
// CharAllocate/16384               5680 ns         5680 ns       120580 bytes_per_second=2.68638Gi/s items_per_second=176.055k/s
// CharAllocate/65536              21925 ns        21923 ns        31785 bytes_per_second=2.78403Gi/s items_per_second=45.6135k/s
// CharAllocate/262144            261386 ns       261282 ns         2678 bytes_per_second=956.821Mi/s items_per_second=3.82728k/s
// CharAllocate/1048576          1274904 ns      1274801 ns          541 bytes_per_second=784.436Mi/s items_per_second=784.436/s

// ShortAllocate/1                  14.1 ns         14.1 ns     49634538 bytes_per_second=135.304Mi/s items_per_second=70.9384M/s
// ShortAllocate/4                  50.1 ns         50.1 ns     13925599 bytes_per_second=152.383Mi/s items_per_second=19.9731M/s
// ShortAllocate/16                 89.7 ns         89.7 ns      7886768 bytes_per_second=340.32Mi/s items_per_second=11.1516M/s
// ShortAllocate/64                  149 ns          149 ns      4681108 bytes_per_second=818.018Mi/s items_per_second=6.7012M/s
// ShortAllocate/256                 292 ns          292 ns      2409484 bytes_per_second=1.63076Gi/s items_per_second=3.41994M/s
// ShortAllocate/1024                754 ns          754 ns       919873 bytes_per_second=2.52874Gi/s items_per_second=1.32579M/s
// ShortAllocate/4096               2394 ns         2394 ns       292148 bytes_per_second=3.18734Gi/s items_per_second=417.771k/s
// ShortAllocate/16384              8787 ns         8787 ns        79341 bytes_per_second=3.47316Gi/s items_per_second=113.809k/s
// ShortAllocate/65536             34309 ns        34306 ns        20407 bytes_per_second=3.55829Gi/s items_per_second=29.1495k/s
// ShortAllocate/262144           136875 ns       136875 ns         5096 bytes_per_second=3.56735Gi/s items_per_second=7.30593k/s
// ShortAllocate/1048576         2598899 ns      2598492 ns          270 bytes_per_second=769.677Mi/s items_per_second=384.839/s

// IntAllocate/1                    12.2 ns         12.2 ns     57787317 bytes_per_second=313.647Mi/s items_per_second=82.2208M/s
// IntAllocate/4                    45.4 ns         45.4 ns     14949173 bytes_per_second=336.298Mi/s items_per_second=22.0396M/s
// IntAllocate/16                   82.9 ns         82.9 ns      8480034 bytes_per_second=735.919Mi/s items_per_second=12.0573M/s
// IntAllocate/64                    128 ns          128 ns      5411735 bytes_per_second=1.85644Gi/s items_per_second=7.78646M/s
// IntAllocate/256                   237 ns          237 ns      2967554 bytes_per_second=4.01691Gi/s items_per_second=4.21203M/s
// IntAllocate/1024                  640 ns          640 ns      1100221 bytes_per_second=5.96377Gi/s items_per_second=1.56337M/s
// IntAllocate/4096                 1801 ns         1800 ns       389484 bytes_per_second=8.4754Gi/s items_per_second=555.444k/s
// IntAllocate/16384                6543 ns         6543 ns       105914 bytes_per_second=9.32804Gi/s items_per_second=152.831k/s
// IntAllocate/65536               26436 ns        26434 ns        26555 bytes_per_second=9.23588Gi/s items_per_second=37.8302k/s
// IntAllocate/262144             107072 ns       107063 ns         6408 bytes_per_second=9.12134Gi/s items_per_second=9.34025k/s
// IntAllocate/1048576           5036275 ns      5034954 ns          126 bytes_per_second=794.446Mi/s items_per_second=198.612/s

// LongLongAllocate/1               12.4 ns         12.4 ns     56339052 bytes_per_second=615.75Mi/s items_per_second=80.7075M/s
// LongLongAllocate/4               47.2 ns         47.2 ns     14970476 bytes_per_second=646.752Mi/s items_per_second=21.1928M/s
// LongLongAllocate/16              85.7 ns         85.7 ns      8017070 bytes_per_second=1.39106Gi/s items_per_second=11.669M/s
// LongLongAllocate/64               143 ns          143 ns      4881281 bytes_per_second=3.32451Gi/s items_per_second=6.97201M/s
// LongLongAllocate/256              335 ns          335 ns      2094763 bytes_per_second=5.69381Gi/s items_per_second=2.9852M/s
// LongLongAllocate/1024             870 ns          870 ns       798891 bytes_per_second=8.76727Gi/s items_per_second=1.14914M/s
// LongLongAllocate/4096            2854 ns         2853 ns       244950 bytes_per_second=10.6954Gi/s items_per_second=350.467k/s
// LongLongAllocate/16384          10521 ns        10521 ns        66354 bytes_per_second=11.6024Gi/s items_per_second=95.0469k/s
// LongLongAllocate/65536          42186 ns        42182 ns        16489 bytes_per_second=11.5755Gi/s items_per_second=23.7066k/s
// LongLongAllocate/262144        282560 ns       282530 ns         2427 bytes_per_second=6.91299Gi/s items_per_second=3.53945k/s
// LongLongAllocate/1048576     11096607 ns     11093783 ns           59 bytes_per_second=721.125Mi/s items_per_second=90.1406/s

// FloatAllocate/1                  12.1 ns         12.1 ns     57573628 bytes_per_second=314.027Mi/s items_per_second=82.3202M/s
// FloatAllocate/4                  45.2 ns         45.2 ns     15574853 bytes_per_second=337.259Mi/s items_per_second=22.1026M/s
// FloatAllocate/16                 82.0 ns         82.0 ns      8550844 bytes_per_second=744.344Mi/s items_per_second=12.1953M/s
// FloatAllocate/64                  130 ns          130 ns      5284563 bytes_per_second=1.83877Gi/s items_per_second=7.71238M/s
// FloatAllocate/256                 235 ns          235 ns      2969189 bytes_per_second=4.06659Gi/s items_per_second=4.26413M/s
// FloatAllocate/1024                625 ns          625 ns      1115550 bytes_per_second=6.10256Gi/s items_per_second=1.59975M/s
// FloatAllocate/4096               1801 ns         1801 ns       387882 bytes_per_second=8.47249Gi/s items_per_second=555.253k/s
// FloatAllocate/16384              6617 ns         6616 ns       105344 bytes_per_second=9.22509Gi/s items_per_second=151.144k/s
// FloatAllocate/65536             26494 ns        26490 ns        26092 bytes_per_second=9.21627Gi/s items_per_second=37.7498k/s
// FloatAllocate/262144           107319 ns       107317 ns         6467 bytes_per_second=9.09983Gi/s items_per_second=9.31823k/s
// FloatAllocate/1048576         1517769 ns      1517433 ns          467 bytes_per_second=2.57425Gi/s items_per_second=659.008/s

// DoubleAllocate/1                 13.9 ns         13.9 ns     54079747 bytes_per_second=549.745Mi/s items_per_second=72.0562M/s
// DoubleAllocate/4                 47.6 ns         47.6 ns     14723020 bytes_per_second=640.909Mi/s items_per_second=21.0013M/s
// DoubleAllocate/16                85.7 ns         85.7 ns      8140536 bytes_per_second=1.39149Gi/s items_per_second=11.6727M/s
// DoubleAllocate/64                 143 ns          143 ns      4855517 bytes_per_second=3.33453Gi/s items_per_second=6.99301M/s
// DoubleAllocate/256                331 ns          331 ns      2119422 bytes_per_second=5.76263Gi/s items_per_second=3.02128M/s
// DoubleAllocate/1024               866 ns          866 ns       809133 bytes_per_second=8.80744Gi/s items_per_second=1.15441M/s
// DoubleAllocate/4096              2832 ns         2832 ns       247065 bytes_per_second=10.7754Gi/s items_per_second=353.09k/s
// DoubleAllocate/16384            10526 ns        10526 ns        65795 bytes_per_second=11.5968Gi/s items_per_second=95.0014k/s
// DoubleAllocate/65536            42413 ns        42406 ns        16504 bytes_per_second=11.5143Gi/s items_per_second=23.5814k/s
// DoubleAllocate/262144          301829 ns       301755 ns         2310 bytes_per_second=6.47256Gi/s items_per_second=3.31395k/s
// DoubleAllocate/1048576       12418406 ns     12414327 ns           59 bytes_per_second=644.417Mi/s items_per_second=80.5521/s

// LongDoubleAllocate/1             14.9 ns         14.9 ns     47338655 bytes_per_second=1.0032Gi/s items_per_second=67.3235M/s
// LongDoubleAllocate/4             46.1 ns         46.1 ns     15163948 bytes_per_second=1.29291Gi/s items_per_second=21.6915M/s
// LongDoubleAllocate/16            84.0 ns         84.0 ns      7825748 bytes_per_second=2.83859Gi/s items_per_second=11.9059M/s
// LongDoubleAllocate/64             173 ns          173 ns      4053110 bytes_per_second=5.50853Gi/s items_per_second=5.77611M/s
// LongDoubleAllocate/256            542 ns          542 ns      1294121 bytes_per_second=7.03713Gi/s items_per_second=1.84474M/s
// LongDoubleAllocate/1024          1708 ns         1708 ns       409803 bytes_per_second=8.93351Gi/s items_per_second=585.467k/s
// LongDoubleAllocate/4096          6424 ns         6423 ns       108194 bytes_per_second=9.5024Gi/s items_per_second=155.687k/s
// LongDoubleAllocate/16384        25417 ns        25414 ns        27711 bytes_per_second=9.60648Gi/s items_per_second=39.3481k/s
// LongDoubleAllocate/65536       103878 ns       103871 ns         6142 bytes_per_second=9.40172Gi/s items_per_second=9.62736k/s
// LongDoubleAllocate/262144     2616307 ns      2615519 ns          433 bytes_per_second=1.49349Gi/s items_per_second=382.333/s
// LongDoubleAllocate/1048576   27988943 ns     27981922 ns           27 bytes_per_second=571.798Mi/s items_per_second=35.7374/s
// CharAllocate
// 15.77 │ 80:   movb     $0x0,(%rax)                                                                                                                                       ▒
//  13.62 │ 83:   inc      %rax                                                                                                                                              ▒
//  14.04 │       dec      %ebx                                                                                                                                              ▒
//  20.27 │    ┌──je       120                                                                                                                                               ▒
//  18.42 │ 8e:│  cmp      %r14,%rax                                                                                                                                         ▒
//   0.09 │    │↑ jne      80 
//   ...
//   vectore::reallocate inline function
// IntAllocate
// 14.52 │ 80:┌─→movl     $0x0,(%rax)                                                                                                                                       ▒
//  13.96 │ 86:│  add      $0x4,%rax                                                                                                                                         ▒
//  14.45 │    │  dec      %ebx                                                                                                                                              ▒
//  21.61 │    │↓ je       130                                                                                                                                               ▒
//  17.83 │ 92:├──cmp      %r14,%rax                                                                                                                                         ▒
//   0.06 │    └──jne      80       
// LongDoubleAllocate
//   2.66 │ 80:┌─→fldz                                                                                                                                                       ▒
//  65.77 │    │  fstpt    (%rax)                                                                                                                                            ▒
//   2.76 │ 84:│  add      $0x10,%rax                                                                                                                                        ▒
//   2.56 │    │  dec      %ebx                                                                                                                                              ▒
//   2.65 │    │↓ je       130                                                                                                                                               ▒
//   4.89 │ 90:├──cmp      %r14,%rax                                                                                                                                         ▒
//   0.01 │    └──jne      80    

// Note1: The clang version implement thet pushback with inline, which make the code much better than the g++ version

BENCHMARK_MAIN();