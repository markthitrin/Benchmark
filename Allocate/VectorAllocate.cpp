#include <benchmark/benchmark.h>
#include <vector>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<char> v(array_size);
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<short> v(array_size);
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<int> v(array_size);
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long long> v(array_size);
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<float> v(array_size);
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<double> v(array_size);
    escape(v.data());
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long double> v(array_size);
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
// CharAllocate/1                   10.3 ns         10.3 ns     68143356 bytes_per_second=92.8216Mi/s items_per_second=97.3305M/s
// CharAllocate/4                   15.7 ns         15.7 ns     44527942 bytes_per_second=242.654Mi/s items_per_second=63.6102M/s
// CharAllocate/16                  15.0 ns         15.0 ns     46613867 bytes_per_second=1017.74Mi/s items_per_second=66.6984M/s
// CharAllocate/64                  14.9 ns         14.9 ns     46741034 bytes_per_second=3.9878Gi/s items_per_second=66.9042M/s
// CharAllocate/256                 15.0 ns         15.0 ns     46729128 bytes_per_second=15.904Gi/s items_per_second=66.7062M/s
// CharAllocate/1024                18.2 ns         18.2 ns     38430393 bytes_per_second=52.3979Gi/s items_per_second=54.9432M/s
// CharAllocate/4096                64.7 ns         64.7 ns     10656969 bytes_per_second=58.9362Gi/s items_per_second=15.4498M/s
// CharAllocate/16384                164 ns          164 ns      4200009 bytes_per_second=93.0303Gi/s items_per_second=6.09683M/s
// CharAllocate/65536                550 ns          550 ns      1270809 bytes_per_second=111.047Gi/s items_per_second=1.81939M/s
// CharAllocate/262144              2118 ns         2118 ns       329779 bytes_per_second=115.272Gi/s items_per_second=472.152k/s
// CharAllocate/1048576            11762 ns        11761 ns        59510 bytes_per_second=83.0374Gi/s items_per_second=85.0303k/s

// ShortAllocate/1                  11.7 ns         11.7 ns     65749873 bytes_per_second=162.593Mi/s items_per_second=85.2458M/s
// ShortAllocate/4                  15.5 ns         15.5 ns     44418223 bytes_per_second=492.063Mi/s items_per_second=64.4957M/s
// ShortAllocate/16                 15.2 ns         15.2 ns     46528173 bytes_per_second=1.95973Gi/s items_per_second=65.7577M/s
// ShortAllocate/64                 15.2 ns         15.2 ns     46107815 bytes_per_second=7.84495Gi/s items_per_second=65.8082M/s
// ShortAllocate/256                15.1 ns         15.1 ns     46335259 bytes_per_second=31.6043Gi/s items_per_second=66.279M/s
// ShortAllocate/1024               53.3 ns         53.3 ns     13031660 bytes_per_second=35.8032Gi/s items_per_second=18.7712M/s
// ShortAllocate/4096               93.1 ns         93.1 ns      7414514 bytes_per_second=81.9699Gi/s items_per_second=10.744M/s
// ShortAllocate/16384               285 ns          285 ns      2458483 bytes_per_second=107.035Gi/s items_per_second=3.50734M/s
// ShortAllocate/65536              1063 ns         1062 ns       653923 bytes_per_second=114.899Gi/s items_per_second=941.25k/s
// ShortAllocate/262144             5340 ns         5340 ns       126314 bytes_per_second=91.4397Gi/s items_per_second=187.269k/s
// ShortAllocate/1048576           25881 ns        25880 ns        26956 bytes_per_second=75.4683Gi/s items_per_second=38.6398k/s

// IntAllocate/1                    10.5 ns         10.5 ns     65859122 bytes_per_second=363.085Mi/s items_per_second=95.1807M/s
// IntAllocate/4                    15.5 ns         15.5 ns     44991809 bytes_per_second=986.777Mi/s items_per_second=64.6694M/s
// IntAllocate/16                   15.2 ns         15.2 ns     46168486 bytes_per_second=3.93043Gi/s items_per_second=65.9417M/s
// IntAllocate/64                   15.2 ns         15.2 ns     45294764 bytes_per_second=15.6416Gi/s items_per_second=65.6058M/s
// IntAllocate/256                  17.6 ns         17.6 ns     39961499 bytes_per_second=54.0944Gi/s items_per_second=56.7221M/s
// IntAllocate/1024                 65.2 ns         65.2 ns     10601848 bytes_per_second=58.4797Gi/s items_per_second=15.3301M/s
// IntAllocate/4096                  164 ns          164 ns      4238399 bytes_per_second=92.8696Gi/s items_per_second=6.0863M/s
// IntAllocate/16384                 549 ns          549 ns      1257102 bytes_per_second=111.207Gi/s items_per_second=1.82202M/s
// IntAllocate/65536                2123 ns         2122 ns       326336 bytes_per_second=115.033Gi/s items_per_second=471.174k/s
// IntAllocate/262144              12124 ns        12123 ns        55718 bytes_per_second=80.556Gi/s items_per_second=82.4894k/s
// IntAllocate/1048576            209494 ns       209477 ns         3815 bytes_per_second=18.6477Gi/s items_per_second=4.7738k/s

// LongLongAllocate/1               10.2 ns         10.2 ns     68392605 bytes_per_second=744.745Mi/s items_per_second=97.6153M/s
// LongLongAllocate/4               15.2 ns         15.2 ns     45848921 bytes_per_second=1.961Gi/s items_per_second=65.8002M/s
// LongLongAllocate/16              15.2 ns         15.2 ns     46129100 bytes_per_second=7.8471Gi/s items_per_second=65.8263M/s
// LongLongAllocate/64              15.1 ns         15.1 ns     46796159 bytes_per_second=31.567Gi/s items_per_second=66.2008M/s
// LongLongAllocate/256             54.1 ns         54.1 ns     12155211 bytes_per_second=35.2489Gi/s items_per_second=18.4806M/s
// LongLongAllocate/1024            93.4 ns         93.4 ns      7485237 bytes_per_second=81.6969Gi/s items_per_second=10.7082M/s
// LongLongAllocate/4096             286 ns          286 ns      2449576 bytes_per_second=106.631Gi/s items_per_second=3.49408M/s
// LongLongAllocate/16384           1060 ns         1060 ns       660520 bytes_per_second=115.174Gi/s items_per_second=943.506k/s
// LongLongAllocate/65536           4896 ns         4895 ns       144326 bytes_per_second=99.7512Gi/s items_per_second=204.29k/s
// LongLongAllocate/262144         24593 ns        24592 ns        28506 bytes_per_second=79.4218Gi/s items_per_second=40.664k/s
// LongLongAllocate/1048576      1494119 ns      1493720 ns          473 bytes_per_second=5.23023Gi/s items_per_second=669.47/s

// FloatAllocate/1                  11.2 ns         11.2 ns     62682933 bytes_per_second=341.041Mi/s items_per_second=89.4018M/s
// FloatAllocate/4                  14.9 ns         14.9 ns     46508569 bytes_per_second=1022.28Mi/s items_per_second=66.9965M/s
// FloatAllocate/16                 14.8 ns         14.8 ns     44379893 bytes_per_second=4.02476Gi/s items_per_second=67.5243M/s
// FloatAllocate/64                 14.9 ns         14.9 ns     47172361 bytes_per_second=16.0368Gi/s items_per_second=67.2633M/s
// FloatAllocate/256                17.6 ns         17.6 ns     39679925 bytes_per_second=54.0723Gi/s items_per_second=56.6989M/s
// FloatAllocate/1024               64.9 ns         64.9 ns     10818368 bytes_per_second=58.7654Gi/s items_per_second=15.405M/s
// FloatAllocate/4096                165 ns          165 ns      4250914 bytes_per_second=92.4556Gi/s items_per_second=6.05917M/s
// FloatAllocate/16384               548 ns          548 ns      1277127 bytes_per_second=111.334Gi/s items_per_second=1.82409M/s
// FloatAllocate/65536              2134 ns         2134 ns       324217 bytes_per_second=114.414Gi/s items_per_second=468.64k/s
// FloatAllocate/262144            12245 ns        12244 ns        56948 bytes_per_second=79.7583Gi/s items_per_second=81.6725k/s
// FloatAllocate/1048576          199162 ns       199137 ns         3856 bytes_per_second=19.6158Gi/s items_per_second=5.02166k/s

// DoubleAllocate/1                 10.5 ns         10.5 ns     66816933 bytes_per_second=727.349Mi/s items_per_second=95.3351M/s
// DoubleAllocate/4                 15.5 ns         15.5 ns     44934276 bytes_per_second=1.9179Gi/s items_per_second=64.354M/s
// DoubleAllocate/16                15.5 ns         15.5 ns     45133401 bytes_per_second=7.6965Gi/s items_per_second=64.563M/s
// DoubleAllocate/64                15.3 ns         15.3 ns     45615358 bytes_per_second=31.1059Gi/s items_per_second=65.2339M/s
// DoubleAllocate/256               55.6 ns         55.6 ns     12922147 bytes_per_second=34.2849Gi/s items_per_second=17.9752M/s
// DoubleAllocate/1024              94.2 ns         94.1 ns      7472916 bytes_per_second=81.0424Gi/s items_per_second=10.6224M/s
// DoubleAllocate/4096               285 ns          285 ns      2444016 bytes_per_second=107.069Gi/s items_per_second=3.50844M/s
// DoubleAllocate/16384             1063 ns         1063 ns       660421 bytes_per_second=114.838Gi/s items_per_second=940.757k/s
// DoubleAllocate/65536             4856 ns         4856 ns       144185 bytes_per_second=100.559Gi/s items_per_second=205.944k/s
// DoubleAllocate/262144           25466 ns        25467 ns        28270 bytes_per_second=76.6938Gi/s items_per_second=39.2672k/s
// DoubleAllocate/1048576        1509099 ns      1508756 ns          468 bytes_per_second=5.17811Gi/s items_per_second=662.798/s

// LongDoubleAllocate/1             11.5 ns         11.5 ns     61106934 bytes_per_second=1.30001Gi/s items_per_second=87.2419M/s
// LongDoubleAllocate/4             14.9 ns         14.9 ns     44147505 bytes_per_second=3.99205Gi/s items_per_second=66.9754M/s
// LongDoubleAllocate/16            14.8 ns         14.8 ns     42470231 bytes_per_second=16.1084Gi/s items_per_second=67.5634M/s
// LongDoubleAllocate/64            16.9 ns         16.9 ns     41543519 bytes_per_second=56.4073Gi/s items_per_second=59.1474M/s
// LongDoubleAllocate/256           66.6 ns         66.6 ns     10530500 bytes_per_second=57.2714Gi/s items_per_second=15.0134M/s
// LongDoubleAllocate/1024           169 ns          169 ns      4208451 bytes_per_second=90.3401Gi/s items_per_second=5.92053M/s
// LongDoubleAllocate/4096           561 ns          561 ns      1214870 bytes_per_second=108.883Gi/s items_per_second=1.78394M/s
// LongDoubleAllocate/16384         2122 ns         2122 ns       332162 bytes_per_second=115.052Gi/s items_per_second=471.254k/s
// LongDoubleAllocate/65536        12492 ns        12489 ns        55802 bytes_per_second=78.1955Gi/s items_per_second=80.0722k/s
// LongDoubleAllocate/262144      349807 ns       349757 ns         3785 bytes_per_second=11.1685Gi/s items_per_second=2.85913k/s
// LongDoubleAllocate/1048576    5455949 ns      5454619 ns          192 bytes_per_second=2.86454Gi/s items_per_second=183.331/s
// CharAllocate
//   7.69 │190:   mov      %r14,%rdi                                                                                                                                         ▒
//  14.33 │     → call     operator new(unsigned long)@plt                                                                                                                   ▒
//   4.11 │       movb     $0x0,(%rax)                                                                                                                                       ▒
//   3.34 │       mov      %rax,%r15                                                                                                                                         ▒
//   4.26 │       cmp      $0x1,%r14                                                                                                                                         ▒
//        │     ↓ je       1b7                                                                                                                                               ◆
//   3.98 │       mov      -0x68(%rbp),%rax                                                                                                                                  ▒
//   3.09 │       lea      0x1(%r15),%rdi                                                                                                                                    ▒
//   2.71 │       xor      %esi,%esi                                                                                                                                         ▒
//   2.81 │       lea      -0x1(%rax),%rdx
//   5.96 │     → call     memset@plt                                                                                                                                        ▒
//   5.27 │1b7:   mov      %r14,%rsi                                                                                                                                         ▒
//   6.87 │       mov      %r15,%rdi                                                                                                                                         ▒
//  13.79 │     → call     operator delete(void*, unsigned long)@plt                                                                                                         ▒
//   6.00 │       test     %r13,%r13                                                                                                                                         ▒
//   2.47 │     ↓ jle      217                                                                                                                                               ▒
//   6.29 │       sub      $0x1,%r13                                                                                                                                         ▒
//   7.02 │     ↑ jne      190


// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   11.1 ns         11.1 ns     63306884 bytes_per_second=86.1625Mi/s items_per_second=90.3479M/s
// CharAllocate/4                   16.4 ns         16.4 ns     42585782 bytes_per_second=232.482Mi/s items_per_second=60.9437M/s
// CharAllocate/16                  15.6 ns         15.6 ns     44632279 bytes_per_second=978.901Mi/s items_per_second=64.1533M/s
// CharAllocate/64                  15.3 ns         15.3 ns     44920975 bytes_per_second=3.90028Gi/s items_per_second=65.4359M/s
// CharAllocate/256                 15.6 ns         15.6 ns     44830088 bytes_per_second=15.26Gi/s items_per_second=64.0051M/s
// CharAllocate/1024                18.6 ns         18.6 ns     37640806 bytes_per_second=51.2422Gi/s items_per_second=53.7313M/s
// CharAllocate/4096                65.2 ns         65.1 ns     10495475 bytes_per_second=58.5643Gi/s items_per_second=15.3523M/s
// CharAllocate/16384                166 ns          166 ns      4217253 bytes_per_second=92.1962Gi/s items_per_second=6.04217M/s
// CharAllocate/65536                553 ns          553 ns      1263774 bytes_per_second=110.37Gi/s items_per_second=1.8083M/s
// CharAllocate/262144              2139 ns         2138 ns       327324 bytes_per_second=114.174Gi/s items_per_second=467.656k/s
// CharAllocate/1048576            11875 ns        11870 ns        58995 bytes_per_second=82.2698Gi/s items_per_second=84.2443k/s

// ShortAllocate/1                  10.8 ns         10.8 ns     64785131 bytes_per_second=176.266Mi/s items_per_second=92.414M/s
// ShortAllocate/4                  15.1 ns         15.1 ns     46566796 bytes_per_second=505.605Mi/s items_per_second=66.2707M/s
// ShortAllocate/16                 14.9 ns         14.9 ns     46720119 bytes_per_second=1.99554Gi/s items_per_second=66.9592M/s
// ShortAllocate/64                 15.2 ns         15.2 ns     46289270 bytes_per_second=7.84224Gi/s items_per_second=65.7855M/s
// ShortAllocate/256                15.3 ns         15.3 ns     46146087 bytes_per_second=31.1934Gi/s items_per_second=65.4173M/s
// ShortAllocate/1024               54.2 ns         54.2 ns     13094909 bytes_per_second=35.2124Gi/s items_per_second=18.4614M/s
// ShortAllocate/4096               94.7 ns         94.7 ns      7391106 bytes_per_second=80.5701Gi/s items_per_second=10.5605M/s
// ShortAllocate/16384               288 ns          288 ns      2440177 bytes_per_second=106.057Gi/s items_per_second=3.47527M/s
// ShortAllocate/65536              1070 ns         1069 ns       653581 bytes_per_second=114.182Gi/s items_per_second=935.377k/s
// ShortAllocate/262144             5267 ns         5265 ns       132195 bytes_per_second=92.7344Gi/s items_per_second=189.92k/s
// ShortAllocate/1048576           26343 ns        26333 ns        26550 bytes_per_second=74.1716Gi/s items_per_second=37.9758k/s

// IntAllocate/1                    10.8 ns         10.8 ns     64304421 bytes_per_second=352.321Mi/s items_per_second=92.3588M/s
// IntAllocate/4                    15.2 ns         15.2 ns     47158171 bytes_per_second=1002.54Mi/s items_per_second=65.7022M/s
// IntAllocate/16                   15.0 ns         15.0 ns     47464231 bytes_per_second=3.96841Gi/s items_per_second=66.5789M/s
// IntAllocate/64                   15.2 ns         15.2 ns     45590311 bytes_per_second=15.7087Gi/s items_per_second=65.8869M/s
// IntAllocate/256                  18.0 ns         18.0 ns     38807269 bytes_per_second=52.9101Gi/s items_per_second=55.4803M/s
// IntAllocate/1024                 71.6 ns         71.6 ns      9683868 bytes_per_second=53.2735Gi/s items_per_second=13.9653M/s
// IntAllocate/4096                  169 ns          169 ns      4155576 bytes_per_second=90.3693Gi/s items_per_second=5.92244M/s
// IntAllocate/16384                 554 ns          554 ns      1264457 bytes_per_second=110.141Gi/s items_per_second=1.80455M/s
// IntAllocate/65536                2141 ns         2140 ns       326685 bytes_per_second=114.082Gi/s items_per_second=467.281k/s
// IntAllocate/262144              12237 ns        12234 ns        58852 bytes_per_second=79.8267Gi/s items_per_second=81.7425k/s
// IntAllocate/1048576            205872 ns       205793 ns         3342 bytes_per_second=18.9815Gi/s items_per_second=4.85925k/s

// LongLongAllocate/1               10.9 ns         10.9 ns     64820819 bytes_per_second=698.26Mi/s items_per_second=91.5223M/s
// LongLongAllocate/4               15.3 ns         15.3 ns     45783272 bytes_per_second=1.95237Gi/s items_per_second=65.5106M/s
// LongLongAllocate/16              15.1 ns         15.1 ns     46094259 bytes_per_second=7.88519Gi/s items_per_second=66.1457M/s
// LongLongAllocate/64              15.2 ns         15.2 ns     45859643 bytes_per_second=31.4067Gi/s items_per_second=65.8645M/s
// LongLongAllocate/256             65.8 ns         65.7 ns     10626179 bytes_per_second=29.0174Gi/s items_per_second=15.2135M/s
// LongLongAllocate/1024             105 ns          105 ns      6673443 bytes_per_second=72.8042Gi/s items_per_second=9.54259M/s
// LongLongAllocate/4096             291 ns          290 ns      2440949 bytes_per_second=105.079Gi/s items_per_second=3.44322M/s
// LongLongAllocate/16384           1073 ns         1072 ns       653104 bytes_per_second=113.824Gi/s items_per_second=932.447k/s
// LongLongAllocate/65536           5296 ns         5295 ns       131846 bytes_per_second=92.2194Gi/s items_per_second=188.865k/s
// LongLongAllocate/262144         26395 ns        26385 ns        26551 bytes_per_second=74.0247Gi/s items_per_second=37.9006k/s
// LongLongAllocate/1048576      1494931 ns      1494157 ns          470 bytes_per_second=5.2287Gi/s items_per_second=669.274/s

// FloatAllocate/1                  10.8 ns         10.8 ns     64786501 bytes_per_second=352.167Mi/s items_per_second=92.3184M/s
// FloatAllocate/4                  15.3 ns         15.3 ns     45593376 bytes_per_second=999.361Mi/s items_per_second=65.4941M/s
// FloatAllocate/16                 15.1 ns         15.1 ns     46438564 bytes_per_second=3.95949Gi/s items_per_second=66.4292M/s
// FloatAllocate/64                 15.0 ns         15.0 ns     46134457 bytes_per_second=15.9282Gi/s items_per_second=66.8077M/s
// FloatAllocate/256                18.5 ns         18.5 ns     37476502 bytes_per_second=51.5832Gi/s items_per_second=54.0889M/s
// FloatAllocate/1024               66.6 ns         66.5 ns     10558768 bytes_per_second=57.328Gi/s items_per_second=15.0282M/s
// FloatAllocate/4096                166 ns          166 ns      4211642 bytes_per_second=92.1031Gi/s items_per_second=6.03607M/s
// FloatAllocate/16384               554 ns          553 ns      1270418 bytes_per_second=110.301Gi/s items_per_second=1.80718M/s
// FloatAllocate/65536              2142 ns         2142 ns       326828 bytes_per_second=113.987Gi/s items_per_second=466.891k/s
// FloatAllocate/262144            11872 ns        11868 ns        58933 bytes_per_second=82.2863Gi/s items_per_second=84.2612k/s
// FloatAllocate/1048576          201720 ns       201648 ns         3466 bytes_per_second=19.3717Gi/s items_per_second=4.95915k/s

// DoubleAllocate/1                 10.8 ns         10.8 ns     64731449 bytes_per_second=704.143Mi/s items_per_second=92.2934M/s
// DoubleAllocate/4                 15.3 ns         15.3 ns     45611945 bytes_per_second=1.94222Gi/s items_per_second=65.17M/s
// DoubleAllocate/16                15.3 ns         15.3 ns     45482809 bytes_per_second=7.7967Gi/s items_per_second=65.4035M/s
// DoubleAllocate/64                15.4 ns         15.4 ns     45381046 bytes_per_second=30.9271Gi/s items_per_second=64.8587M/s
// DoubleAllocate/256               55.9 ns         55.9 ns     12162223 bytes_per_second=34.13Gi/s items_per_second=17.8939M/s
// DoubleAllocate/1024              95.1 ns         95.0 ns      7366250 bytes_per_second=80.2855Gi/s items_per_second=10.5232M/s
// DoubleAllocate/4096               286 ns          286 ns      2443724 bytes_per_second=106.627Gi/s items_per_second=3.49397M/s
// DoubleAllocate/16384             1072 ns         1072 ns       652682 bytes_per_second=113.873Gi/s items_per_second=932.851k/s
// DoubleAllocate/65536             5282 ns         5281 ns       131409 bytes_per_second=92.4676Gi/s items_per_second=189.374k/s
// DoubleAllocate/262144           26385 ns        26377 ns        26493 bytes_per_second=74.0467Gi/s items_per_second=37.9119k/s
// DoubleAllocate/1048576        1494065 ns      1493390 ns          448 bytes_per_second=5.23139Gi/s items_per_second=669.618/s

// LongDoubleAllocate/1             12.1 ns         12.1 ns     57963545 bytes_per_second=1.23214Gi/s items_per_second=82.6874M/s
// LongDoubleAllocate/4             19.8 ns         19.8 ns     35667034 bytes_per_second=3.01765Gi/s items_per_second=50.6278M/s
// LongDoubleAllocate/16            34.5 ns         34.5 ns     20480512 bytes_per_second=6.9109Gi/s items_per_second=28.9864M/s
// LongDoubleAllocate/64            92.9 ns         92.9 ns      7527690 bytes_per_second=10.2693Gi/s items_per_second=10.7681M/s
// LongDoubleAllocate/256            377 ns          376 ns      1872938 bytes_per_second=10.1323Gi/s items_per_second=2.65613M/s
// LongDoubleAllocate/1024          1309 ns         1309 ns       532326 bytes_per_second=11.6597Gi/s items_per_second=764.131k/s
// LongDoubleAllocate/4096          5078 ns         5076 ns       136863 bytes_per_second=12.0239Gi/s items_per_second=196.999k/s
// LongDoubleAllocate/16384        20329 ns        20323 ns        34668 bytes_per_second=12.013Gi/s items_per_second=49.2054k/s
// LongDoubleAllocate/65536        83037 ns        83016 ns         8255 bytes_per_second=11.7636Gi/s items_per_second=12.0459k/s
// LongDoubleAllocate/262144      947660 ns       947286 ns         1163 bytes_per_second=4.12362Gi/s items_per_second=1.05565k/s
// LongDoubleAllocate/1048576    5321642 ns      5319805 ns          212 bytes_per_second=2.93714Gi/s items_per_second=187.977/s
// CharAllocate
//  4.84 │ 60:   test     %rbx,%rbx                                                                                                                                         ▒
//        │     ↓ je       a0                                                                                                                                                ◆
//   4.73 │       mov      %r14,%rdi                                                                                                                                         ▒
//  13.01 │     → call     operator new(unsigned long)@plt                                                                                                                   ▒
//   7.40 │       mov      %rax,%r12                                                                                                                                         ▒
//   3.44 │       movb     $0x0,(%rax)                                                                                                                                       ▒
//   3.12 │       test     %r15,%r15                                                                                                                                         ▒
//        │     ↓ je       87                                                                                                                                                ▒
//   1.88 │       lea      0x1(%r12),%rdi                                                                                                                                    ▒
//   3.40 │       xor      %esi,%esi                                                                                                                                         ▒
//   1.96 │       mov      %r15,%rdx
//  10.45 │     → call     memset@plt                                                                                                                                        ▒
//   3.13 │ 87:   mov      %r12,-0x70(%rbp)                                                                                                                                  ▒
//   5.08 │       mov      %r12,%rdi                                                                                                                                         ▒
//   3.17 │       mov      %r14,%rsi                                                                                                                                         ▒
//  11.41 │     → call     operator delete(void*, unsigned long)@plt                                                                                                         ▒
//   6.96 │       test     %r13,%r13                                                                                                                                         ◆
//   3.44 │     ↓ jg       a9                                                                                                                                                ▒
//        │     ↓ jmp      209                                                                                                                                               ▒
//        │ a0:   test     %r13,%r13                                                                                                                                         ▒
//        │     ↓ jle      209                                                                                                                                               ▒
//   7.11 │ a9:   dec      %r13                                                                                                                                              ▒
//   5.48 │     ↑ jne      60 

// Note1: Both g++ and clang++ give the same reseult. it use new operator with memset.

BENCHMARK_MAIN();