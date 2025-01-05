#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = new char[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    short* p = new short[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    int* p = new int[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long long* p = new long long[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    float* p = new float[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    double* p = new double[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long double* p = new long double[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
}

static void Allocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = new char[array_size];
    escape(p);
    delete[] p;
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(Allocate)->RangeMultiplier(2)->Range(1,1<<20);


// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   10.7 ns         10.7 ns     65668970 bytes_per_second=88.8406Mi/s items_per_second=93.1561M/s
// CharAllocate/4                   10.7 ns         10.7 ns     65872304 bytes_per_second=356.826Mi/s items_per_second=93.5399M/s
// CharAllocate/16                  10.4 ns         10.4 ns     67447798 bytes_per_second=1.43755Gi/s items_per_second=96.4722M/s
// CharAllocate/64                  10.4 ns         10.4 ns     67421309 bytes_per_second=5.74943Gi/s items_per_second=96.4594M/s
// CharAllocate/256                 10.4 ns         10.4 ns     66872788 bytes_per_second=22.9884Gi/s items_per_second=96.4204M/s
// CharAllocate/1024                10.5 ns         10.5 ns     66873319 bytes_per_second=91.2519Gi/s items_per_second=95.6845M/s
// CharAllocate/4096                29.4 ns         29.4 ns     24413236 bytes_per_second=129.866Gi/s items_per_second=34.0436M/s
// CharAllocate/16384               28.7 ns         28.7 ns     25428039 bytes_per_second=532.566Gi/s items_per_second=34.9022M/s
// CharAllocate/65536               27.8 ns         27.8 ns     25830996 bytes_per_second=2.14753Ti/s items_per_second=36.0295M/s
// CharAllocate/262144              27.9 ns         27.9 ns     24464860 bytes_per_second=8.55123Ti/s items_per_second=35.8665M/s
// CharAllocate/1048576             28.4 ns         28.4 ns     25882441 bytes_per_second=33.5727Ti/s items_per_second=35.2035M/s

// ShortAllocate/1                  10.6 ns         10.6 ns     65903412 bytes_per_second=179.527Mi/s items_per_second=94.124M/s
// ShortAllocate/4                  10.6 ns         10.6 ns     65493822 bytes_per_second=718.749Mi/s items_per_second=94.2079M/s
// ShortAllocate/16                 10.4 ns         10.4 ns     67403058 bytes_per_second=2.87303Gi/s items_per_second=96.4027M/s
// ShortAllocate/64                 10.3 ns         10.3 ns     67254065 bytes_per_second=11.5476Gi/s items_per_second=96.8684M/s
// ShortAllocate/256                10.2 ns         10.2 ns     68387377 bytes_per_second=46.6481Gi/s items_per_second=97.8282M/s
// ShortAllocate/1024               27.4 ns         27.4 ns     24248579 bytes_per_second=69.6457Gi/s items_per_second=36.5144M/s
// ShortAllocate/4096               29.0 ns         29.0 ns     24195703 bytes_per_second=263.124Gi/s items_per_second=34.4882M/s
// ShortAllocate/16384              29.3 ns         29.3 ns     24026047 bytes_per_second=1.01797Ti/s items_per_second=34.1575M/s
// ShortAllocate/65536              28.7 ns         28.7 ns     25035295 bytes_per_second=4.15409Ti/s items_per_second=34.8471M/s
// ShortAllocate/262144             29.9 ns         29.9 ns     25515546 bytes_per_second=15.9552Ti/s items_per_second=33.4604M/s
// ShortAllocate/1048576            27.6 ns         27.6 ns     24965045 bytes_per_second=69.2287Ti/s items_per_second=36.2958M/s

// IntAllocate/1                    10.6 ns         10.6 ns     66315680 bytes_per_second=359.025Mi/s items_per_second=94.1161M/s
// IntAllocate/4                    10.3 ns         10.3 ns     67707266 bytes_per_second=1.44678Gi/s items_per_second=97.0917M/s
// IntAllocate/16                   10.3 ns         10.3 ns     67917232 bytes_per_second=5.78932Gi/s items_per_second=97.1286M/s
// IntAllocate/64                   10.3 ns         10.3 ns     67855496 bytes_per_second=23.1719Gi/s items_per_second=97.1898M/s
// IntAllocate/256                  10.3 ns         10.3 ns     67471773 bytes_per_second=92.4868Gi/s items_per_second=96.9795M/s
// IntAllocate/1024                 28.4 ns         28.4 ns     26309770 bytes_per_second=134.474Gi/s items_per_second=35.2516M/s
// IntAllocate/4096                 29.8 ns         29.8 ns     26240087 bytes_per_second=511.51Gi/s items_per_second=33.5223M/s
// IntAllocate/16384                26.9 ns         26.8 ns     23385799 bytes_per_second=2.22015Ti/s items_per_second=37.248M/s
// IntAllocate/65536                28.2 ns         28.2 ns     26098086 bytes_per_second=8.46489Ti/s items_per_second=35.5043M/s
// IntAllocate/262144               27.3 ns         27.3 ns     24056491 bytes_per_second=34.932Ti/s items_per_second=36.6289M/s
// IntAllocate/1048576              28.2 ns         28.2 ns     22002081 bytes_per_second=135.165Ti/s items_per_second=35.4327M/s

// LongLongAllocate/1               10.5 ns         10.5 ns     65664342 bytes_per_second=726.206Mi/s items_per_second=95.1853M/s
// LongLongAllocate/4               10.3 ns         10.3 ns     68119213 bytes_per_second=2.90488Gi/s items_per_second=97.4717M/s
// LongLongAllocate/16              10.3 ns         10.3 ns     68145536 bytes_per_second=11.6143Gi/s items_per_second=97.4276M/s
// LongLongAllocate/64              10.3 ns         10.3 ns     67919476 bytes_per_second=46.4571Gi/s items_per_second=97.4275M/s
// LongLongAllocate/256             29.5 ns         29.5 ns     25599625 bytes_per_second=64.7569Gi/s items_per_second=33.9513M/s
// LongLongAllocate/1024            28.6 ns         28.6 ns     23607271 bytes_per_second=266.882Gi/s items_per_second=34.9808M/s
// LongLongAllocate/4096            28.8 ns         28.8 ns     25181526 bytes_per_second=1.0335Ti/s items_per_second=34.6786M/s
// LongLongAllocate/16384           30.0 ns         30.0 ns     23264012 bytes_per_second=3.97673Ti/s items_per_second=33.3592M/s
// LongLongAllocate/65536           28.6 ns         28.6 ns     24764086 bytes_per_second=16.6969Ti/s items_per_second=35.016M/s
// LongLongAllocate/262144          29.1 ns         29.1 ns     24622802 bytes_per_second=65.5993Ti/s items_per_second=34.3929M/s
// LongLongAllocate/1048576         29.6 ns         29.6 ns     23671883 bytes_per_second=257.524Ti/s items_per_second=33.7542M/s

// FloatAllocate/1                  10.5 ns         10.4 ns     66836638 bytes_per_second=365.064Mi/s items_per_second=95.6993M/s
// FloatAllocate/4                  10.2 ns         10.2 ns     68265839 bytes_per_second=1.45926Gi/s items_per_second=97.9292M/s
// FloatAllocate/16                 10.2 ns         10.2 ns     68327890 bytes_per_second=5.83739Gi/s items_per_second=97.9352M/s
// FloatAllocate/64                 10.2 ns         10.2 ns     68335562 bytes_per_second=23.3395Gi/s items_per_second=97.893M/s
// FloatAllocate/256                10.3 ns         10.3 ns     67922706 bytes_per_second=92.3024Gi/s items_per_second=96.7861M/s
// FloatAllocate/1024               29.5 ns         29.5 ns     24107613 bytes_per_second=129.26Gi/s items_per_second=33.8848M/s
// FloatAllocate/4096               28.6 ns         28.6 ns     25991370 bytes_per_second=533.996Gi/s items_per_second=34.996M/s
// FloatAllocate/16384              29.5 ns         29.5 ns     24051829 bytes_per_second=2.02062Ti/s items_per_second=33.9004M/s
// FloatAllocate/65536              28.3 ns         28.3 ns     25254356 bytes_per_second=8.42725Ti/s items_per_second=35.3464M/s
// FloatAllocate/262144             27.5 ns         27.5 ns     25480313 bytes_per_second=34.6978Ti/s items_per_second=36.3833M/s
// FloatAllocate/1048576            27.2 ns         27.2 ns     26232984 bytes_per_second=140.123Ti/s items_per_second=36.7324M/s

// DoubleAllocate/1                 10.5 ns         10.4 ns     66806954 bytes_per_second=730.187Mi/s items_per_second=95.7071M/s
// DoubleAllocate/4                 10.2 ns         10.2 ns     68353793 bytes_per_second=2.91716Gi/s items_per_second=97.8838M/s
// DoubleAllocate/16                10.2 ns         10.2 ns     68412467 bytes_per_second=11.6715Gi/s items_per_second=97.9078M/s
// DoubleAllocate/64                10.2 ns         10.2 ns     68350857 bytes_per_second=46.6513Gi/s items_per_second=97.8349M/s
// DoubleAllocate/256               28.8 ns         28.8 ns     24683606 bytes_per_second=66.1608Gi/s items_per_second=34.6873M/s
// DoubleAllocate/1024              27.9 ns         27.9 ns     25186313 bytes_per_second=273.64Gi/s items_per_second=35.8665M/s
// DoubleAllocate/4096              28.0 ns         28.0 ns     24919442 bytes_per_second=1.06409Ti/s items_per_second=35.7049M/s
// DoubleAllocate/16384             28.1 ns         28.1 ns     25344253 bytes_per_second=4.23948Ti/s items_per_second=35.5634M/s
// DoubleAllocate/65536             29.4 ns         29.4 ns     23474642 bytes_per_second=16.1957Ti/s items_per_second=33.9649M/s
// DoubleAllocate/262144            28.9 ns         28.9 ns     24963968 bytes_per_second=65.9614Ti/s items_per_second=34.5828M/s
// DoubleAllocate/1048576           29.5 ns         29.5 ns     24648318 bytes_per_second=258.748Ti/s items_per_second=33.9147M/s

// LongDoubleAllocate/1             10.3 ns         10.3 ns     68132950 bytes_per_second=1.45063Gi/s items_per_second=97.3499M/s
// LongDoubleAllocate/4             10.3 ns         10.3 ns     68092659 bytes_per_second=5.79654Gi/s items_per_second=97.2498M/s
// LongDoubleAllocate/16            10.3 ns         10.3 ns     67731526 bytes_per_second=23.1632Gi/s items_per_second=97.1533M/s
// LongDoubleAllocate/64            10.3 ns         10.3 ns     67952670 bytes_per_second=92.7348Gi/s items_per_second=97.2395M/s
// LongDoubleAllocate/256           27.1 ns         27.1 ns     25931697 bytes_per_second=140.657Gi/s items_per_second=36.8724M/s
// LongDoubleAllocate/1024          28.6 ns         28.6 ns     25758483 bytes_per_second=532.815Gi/s items_per_second=34.9185M/s
// LongDoubleAllocate/4096          29.8 ns         29.8 ns     24492321 bytes_per_second=2.00266Ti/s items_per_second=33.5991M/s
// LongDoubleAllocate/16384         26.5 ns         26.5 ns     26710402 bytes_per_second=9.00867Ti/s items_per_second=37.7851M/s
// LongDoubleAllocate/65536         27.5 ns         27.5 ns     24740976 bytes_per_second=34.6875Ti/s items_per_second=36.3725M/s
// LongDoubleAllocate/262144        27.6 ns         27.6 ns     24720517 bytes_per_second=138.155Ti/s items_per_second=36.2164M/s
// LongDoubleAllocate/1048576       26.9 ns         26.9 ns     26376127 bytes_per_second=568.128Ti/s items_per_second=37.2328M/s

// Allocate/1                       10.5 ns         10.4 ns     66381329 bytes_per_second=91.2683Mi/s items_per_second=95.7017M/s
// Allocate/2                       10.6 ns         10.6 ns     66657530 bytes_per_second=180.409Mi/s items_per_second=94.5865M/s
// Allocate/4                       10.5 ns         10.5 ns     66424380 bytes_per_second=362.435Mi/s items_per_second=95.0102M/s
// Allocate/8                       10.5 ns         10.5 ns     66494934 bytes_per_second=724.848Mi/s items_per_second=95.0073M/s
// Allocate/16                      10.3 ns         10.3 ns     65249791 bytes_per_second=1.44854Gi/s items_per_second=97.2101M/s
// Allocate/32                      10.3 ns         10.3 ns     67766387 bytes_per_second=2.89795Gi/s items_per_second=97.2391M/s
// Allocate/64                      10.3 ns         10.3 ns     67549958 bytes_per_second=5.79677Gi/s items_per_second=97.2536M/s
// Allocate/128                     10.3 ns         10.3 ns     67934626 bytes_per_second=11.6123Gi/s items_per_second=97.4113M/s
// Allocate/256                     10.3 ns         10.3 ns     68004091 bytes_per_second=23.233Gi/s items_per_second=97.4463M/s
// Allocate/512                     10.3 ns         10.3 ns     67724756 bytes_per_second=46.4804Gi/s items_per_second=97.4764M/s
// Allocate/1024                    10.3 ns         10.3 ns     67597684 bytes_per_second=92.5892Gi/s items_per_second=97.0868M/s
// Allocate/2048                    29.6 ns         29.6 ns     25736943 bytes_per_second=64.4722Gi/s items_per_second=33.802M/s
// Allocate/4096                    30.7 ns         30.7 ns     25244452 bytes_per_second=124.169Gi/s items_per_second=32.5503M/s
// Allocate/8192                    30.1 ns         30.1 ns     25404830 bytes_per_second=253.152Gi/s items_per_second=33.1812M/s
// Allocate/16384                   29.0 ns         29.0 ns     25156289 bytes_per_second=525.667Gi/s items_per_second=34.4501M/s
// Allocate/32768                   29.1 ns         29.1 ns     24796054 bytes_per_second=1.02526Ti/s items_per_second=34.402M/s
// Allocate/65536                   29.3 ns         29.3 ns     24316739 bytes_per_second=2.0326Ti/s items_per_second=34.1013M/s
// Allocate/131072                  29.3 ns         29.3 ns     24313225 bytes_per_second=4.06825Ti/s items_per_second=34.127M/s
// Allocate/262144                  28.7 ns         28.7 ns     25071860 bytes_per_second=8.30255Ti/s items_per_second=34.8234M/s
// Allocate/524288                  27.9 ns         27.9 ns     23716118 bytes_per_second=17.0751Ti/s items_per_second=35.809M/s
// Allocate/1048576                 26.6 ns         26.6 ns     24904215 bytes_per_second=35.8594Ti/s items_per_second=37.6013M/s

// IntAllocate
//  11.03 │ 88:┌─→mov      %r13,%rdi                                                                                                                                         ◆
//  18.99 │    │→ call     operator new[](unsigned long)@plt                                                                                                                 ▒
//  14.58 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  31.14 │    │→ call     operator delete[](void*)@plt                                                                                                                      ▒
//   9.72 │ 98:│  sub      $0x1,%r12                                                                                                                                         ▒
//  14.53 │    └──jne      88        
// new
//   4.78 │      endbr64                                                                                                                                                     ▒
//   4.70 │      push    %rbp                                                                                                                                                ▒
//   4.46 │      mov     $0x1,%eax                                                                                                                                           ◆
//   9.32 │      mov     %rsp,%rbp                                                                                                                                           ▒
//   4.62 │      push    %rbx                                                                                                                                                ▒
//   7.20 │      sub     $0x8,%rsp                                                                                                                                           ▒
//   3.33 │      test    %rdi,%rdi                                                                                                                                           ▒
//   3.09 │      cmovne  %rdi,%rax                                                                                                                                           ▒
//   3.11 │      mov     %rax,%rbx                                                                                                                                           ▒
//   3.11 │1c:   mov     %rbx,%rdi                                                                                                                                           ▒
//  20.18 │    → call    malloc@plt                                                                                                                                          ▒
//   5.71 │      test    %rax,%rax                                                                                                                                           ▒
//   2.66 │    ↓ je      2f                                                                                                                                                  ▒
//   4.34 │      mov     -0x8(%rbp),%rbx                                                                                                                                     ▒
//  12.38 │      leave
// malloc
// │     → call    _int_malloc
// delete
// 32.35 │      endbr64                                                                                                                                                     ▒
// 67.65 │    → jmp     *0x6afe6(%rip)        # 71e20 <operator delete[](void*)@GLIBCXX_3.4>





// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   11.0 ns         11.0 ns     59160347 bytes_per_second=86.8505Mi/s items_per_second=91.0693M/s
// CharAllocate/4                   10.8 ns         10.8 ns     64406125 bytes_per_second=352.632Mi/s items_per_second=92.4403M/s
// CharAllocate/16                  10.7 ns         10.7 ns     65857514 bytes_per_second=1.3897Gi/s items_per_second=93.2613M/s
// CharAllocate/64                  10.7 ns         10.7 ns     65441614 bytes_per_second=5.5555Gi/s items_per_second=93.2059M/s
// CharAllocate/256                 10.6 ns         10.6 ns     65721356 bytes_per_second=22.4455Gi/s items_per_second=94.1431M/s
// CharAllocate/1024                11.0 ns         11.0 ns     65798998 bytes_per_second=86.401Gi/s items_per_second=90.5981M/s
// CharAllocate/4096                27.0 ns         27.0 ns     25924742 bytes_per_second=141.236Gi/s items_per_second=37.0241M/s
// CharAllocate/16384               30.6 ns         30.6 ns     25423603 bytes_per_second=499.207Gi/s items_per_second=32.716M/s
// CharAllocate/65536               29.5 ns         29.5 ns     24507975 bytes_per_second=2.02378Ti/s items_per_second=33.9534M/s
// CharAllocate/262144              27.1 ns         27.1 ns     25539527 bytes_per_second=8.79437Ti/s items_per_second=36.8863M/s
// CharAllocate/1048576             28.8 ns         28.8 ns     26159861 bytes_per_second=33.1395Ti/s items_per_second=34.7493M/s

// ShortAllocate/1                  10.8 ns         10.8 ns     65871807 bytes_per_second=176.067Mi/s items_per_second=92.3099M/s
// ShortAllocate/4                  10.6 ns         10.6 ns     65858576 bytes_per_second=721.569Mi/s items_per_second=94.5775M/s
// ShortAllocate/16                 10.3 ns         10.3 ns     67098131 bytes_per_second=2.88317Gi/s items_per_second=96.7432M/s
// ShortAllocate/64                 10.3 ns         10.3 ns     67242091 bytes_per_second=11.5326Gi/s items_per_second=96.7427M/s
// ShortAllocate/256                10.3 ns         10.3 ns     67224883 bytes_per_second=46.1154Gi/s items_per_second=96.7111M/s
// ShortAllocate/1024               29.1 ns         29.1 ns     25523114 bytes_per_second=65.631Gi/s items_per_second=34.4095M/s
// ShortAllocate/4096               27.4 ns         27.4 ns     24371185 bytes_per_second=278.839Gi/s items_per_second=36.548M/s
// ShortAllocate/16384              30.4 ns         30.4 ns     23083755 bytes_per_second=1005.06Gi/s items_per_second=32.9337M/s
// ShortAllocate/65536              29.8 ns         29.8 ns     23717256 bytes_per_second=3.99649Ti/s items_per_second=33.525M/s
// ShortAllocate/262144             30.4 ns         30.4 ns     22995538 bytes_per_second=15.663Ti/s items_per_second=32.8477M/s
// ShortAllocate/1048576            27.5 ns         27.5 ns     24425087 bytes_per_second=69.3164Ti/s items_per_second=36.3418M/s

// IntAllocate/1                    10.6 ns         10.6 ns     65937520 bytes_per_second=360.888Mi/s items_per_second=94.6047M/s
// IntAllocate/4                    10.3 ns         10.3 ns     67697478 bytes_per_second=1.4415Gi/s items_per_second=96.7372M/s
// IntAllocate/16                   10.3 ns         10.3 ns     67728616 bytes_per_second=5.76507Gi/s items_per_second=96.7217M/s
// IntAllocate/64                   10.3 ns         10.3 ns     67673999 bytes_per_second=23.0663Gi/s items_per_second=96.7469M/s
// IntAllocate/256                  10.3 ns         10.3 ns     67426065 bytes_per_second=92.2671Gi/s items_per_second=96.7491M/s
// IntAllocate/1024                 26.8 ns         26.8 ns     25700935 bytes_per_second=142.538Gi/s items_per_second=37.3655M/s
// IntAllocate/4096                 26.8 ns         26.8 ns     25634294 bytes_per_second=569.016Gi/s items_per_second=37.291M/s
// IntAllocate/16384                28.6 ns         28.6 ns     24378850 bytes_per_second=2.08665Ti/s items_per_second=35.0082M/s
// IntAllocate/65536                28.0 ns         28.0 ns     24654266 bytes_per_second=8.51938Ti/s items_per_second=35.7329M/s
// IntAllocate/262144               28.0 ns         28.0 ns     26436213 bytes_per_second=34.1118Ti/s items_per_second=35.7688M/s
// IntAllocate/1048576              26.5 ns         26.5 ns     26377152 bytes_per_second=143.726Ti/s items_per_second=37.6769M/s

// LongLongAllocate/1               10.8 ns         10.8 ns     63931037 bytes_per_second=704.882Mi/s items_per_second=92.3903M/s
// LongLongAllocate/4               10.6 ns         10.6 ns     63916948 bytes_per_second=2.81755Gi/s items_per_second=94.5412M/s
// LongLongAllocate/16              10.6 ns         10.6 ns     66167733 bytes_per_second=11.2697Gi/s items_per_second=94.5371M/s
// LongLongAllocate/64              10.6 ns         10.6 ns     66138415 bytes_per_second=45.0358Gi/s items_per_second=94.447M/s
// LongLongAllocate/256             27.5 ns         27.5 ns     25359782 bytes_per_second=69.4056Gi/s items_per_second=36.3885M/s
// LongLongAllocate/1024            28.4 ns         28.4 ns     24248652 bytes_per_second=269.004Gi/s items_per_second=35.2588M/s
// LongLongAllocate/4096            29.2 ns         29.2 ns     24542273 bytes_per_second=1.01958Ti/s items_per_second=34.2116M/s
// LongLongAllocate/16384           29.8 ns         29.7 ns     23640912 bytes_per_second=4.00723Ti/s items_per_second=33.6151M/s
// LongLongAllocate/65536           27.8 ns         27.8 ns     24175972 bytes_per_second=17.1615Ti/s items_per_second=35.9903M/s
// LongLongAllocate/262144          28.4 ns         28.4 ns     24958503 bytes_per_second=67.1538Ti/s items_per_second=35.2079M/s
// LongLongAllocate/1048576         27.2 ns         27.2 ns     25726311 bytes_per_second=280.285Ti/s items_per_second=36.7375M/s

// FloatAllocate/1                  10.8 ns         10.8 ns     64742594 bytes_per_second=353.038Mi/s items_per_second=92.5467M/s
// FloatAllocate/4                  10.6 ns         10.6 ns     66164488 bytes_per_second=1.40961Gi/s items_per_second=94.5976M/s
// FloatAllocate/16                 10.6 ns         10.6 ns     66149203 bytes_per_second=5.63897Gi/s items_per_second=94.6062M/s
// FloatAllocate/64                 10.6 ns         10.6 ns     61885872 bytes_per_second=22.426Gi/s items_per_second=94.0613M/s
// FloatAllocate/256                10.6 ns         10.6 ns     66200885 bytes_per_second=90.1525Gi/s items_per_second=94.5318M/s
// FloatAllocate/1024               28.4 ns         28.4 ns     24945951 bytes_per_second=134.464Gi/s items_per_second=35.2488M/s
// FloatAllocate/4096               27.3 ns         27.3 ns     25825935 bytes_per_second=558.198Gi/s items_per_second=36.582M/s
// FloatAllocate/16384              29.1 ns         29.1 ns     24065248 bytes_per_second=2.04901Ti/s items_per_second=34.3768M/s
// FloatAllocate/65536              28.8 ns         28.8 ns     24569166 bytes_per_second=8.26729Ti/s items_per_second=34.6755M/s
// FloatAllocate/262144             30.0 ns         30.0 ns     24238615 bytes_per_second=31.8095Ti/s items_per_second=33.3546M/s
// FloatAllocate/1048576            27.4 ns         27.4 ns     25162619 bytes_per_second=139.258Ti/s items_per_second=36.5057M/s

// DoubleAllocate/1                 10.5 ns         10.5 ns     66439830 bytes_per_second=725.88Mi/s items_per_second=95.1426M/s
// DoubleAllocate/4                 10.3 ns         10.3 ns     67497929 bytes_per_second=2.90145Gi/s items_per_second=97.3566M/s
// DoubleAllocate/16                10.3 ns         10.3 ns     67577174 bytes_per_second=11.5984Gi/s items_per_second=97.2941M/s
// DoubleAllocate/64                10.3 ns         10.3 ns     67862455 bytes_per_second=46.3401Gi/s items_per_second=97.1822M/s
// DoubleAllocate/256               29.4 ns         29.4 ns     25301619 bytes_per_second=64.8342Gi/s items_per_second=33.9918M/s
// DoubleAllocate/1024              30.0 ns         30.0 ns     24659965 bytes_per_second=254.14Gi/s items_per_second=33.3107M/s
// DoubleAllocate/4096              27.9 ns         27.9 ns     24778674 bytes_per_second=1.06751Ti/s items_per_second=35.8197M/s
// DoubleAllocate/16384             27.7 ns         27.7 ns     24637550 bytes_per_second=4.30452Ti/s items_per_second=36.109M/s
// DoubleAllocate/65536             30.0 ns         30.0 ns     25814567 bytes_per_second=15.9072Ti/s items_per_second=33.3599M/s
// DoubleAllocate/262144            29.7 ns         29.7 ns     22444711 bytes_per_second=64.2598Ti/s items_per_second=33.6906M/s
// DoubleAllocate/1048576           27.2 ns         27.2 ns     23882938 bytes_per_second=280.186Ti/s items_per_second=36.7245M/s

// LongDoubleAllocate/1             10.3 ns         10.3 ns     68111311 bytes_per_second=1.45331Gi/s items_per_second=97.5301M/s
// LongDoubleAllocate/4             10.2 ns         10.2 ns     68182995 bytes_per_second=5.82466Gi/s items_per_second=97.7216M/s
// LongDoubleAllocate/16            10.2 ns         10.2 ns     68143189 bytes_per_second=23.3045Gi/s items_per_second=97.7461M/s
// LongDoubleAllocate/64            10.2 ns         10.2 ns     68126273 bytes_per_second=93.2088Gi/s items_per_second=97.7365M/s
// LongDoubleAllocate/256           29.8 ns         29.8 ns     24096546 bytes_per_second=128.031Gi/s items_per_second=33.5625M/s
// LongDoubleAllocate/1024          30.2 ns         30.2 ns     22504815 bytes_per_second=504.707Gi/s items_per_second=33.0765M/s
// LongDoubleAllocate/4096          29.6 ns         29.6 ns     23151270 bytes_per_second=2.01198Ti/s items_per_second=33.7554M/s
// LongDoubleAllocate/16384         29.9 ns         29.9 ns     24403037 bytes_per_second=7.98707Ti/s items_per_second=33.5002M/s
// LongDoubleAllocate/65536         30.9 ns         30.9 ns     23056675 bytes_per_second=30.8356Ti/s items_per_second=32.3335M/s
// LongDoubleAllocate/262144        28.3 ns         28.3 ns     24082042 bytes_per_second=134.968Ti/s items_per_second=35.381M/s
// LongDoubleAllocate/1048576       30.8 ns         30.8 ns     24515659 bytes_per_second=494.994Ti/s items_per_second=32.4399M/s

// Allocate/1                       10.5 ns         10.5 ns     66556400 bytes_per_second=90.7738Mi/s items_per_second=95.1833M/s
// Allocate/2                       10.5 ns         10.5 ns     66542398 bytes_per_second=181.17Mi/s items_per_second=94.9851M/s
// Allocate/4                       10.5 ns         10.5 ns     66493359 bytes_per_second=363.084Mi/s items_per_second=95.1802M/s
// Allocate/8                       10.6 ns         10.6 ns     66519123 bytes_per_second=723.132Mi/s items_per_second=94.7824M/s
// Allocate/16                      10.3 ns         10.3 ns     67927460 bytes_per_second=1.45147Gi/s items_per_second=97.4068M/s
// Allocate/32                      10.3 ns         10.3 ns     67951685 bytes_per_second=2.89864Gi/s items_per_second=97.2621M/s
// Allocate/64                      10.3 ns         10.3 ns     67941668 bytes_per_second=5.79501Gi/s items_per_second=97.2242M/s
// Allocate/128                     10.3 ns         10.3 ns     67309180 bytes_per_second=11.6045Gi/s items_per_second=97.3456M/s
// Allocate/256                     10.3 ns         10.3 ns     67572372 bytes_per_second=23.1095Gi/s items_per_second=96.9281M/s
// Allocate/512                     10.3 ns         10.3 ns     67632154 bytes_per_second=46.4634Gi/s items_per_second=97.4407M/s
// Allocate/1024                    10.2 ns         10.2 ns     67896819 bytes_per_second=93.1038Gi/s items_per_second=97.6264M/s
// Allocate/2048                    28.3 ns         28.3 ns     25207553 bytes_per_second=67.4316Gi/s items_per_second=35.3536M/s
// Allocate/4096                    27.5 ns         27.5 ns     25247605 bytes_per_second=138.642Gi/s items_per_second=36.3442M/s
// Allocate/8192                    28.2 ns         28.2 ns     26242544 bytes_per_second=270.811Gi/s items_per_second=35.4957M/s
// Allocate/16384                   28.8 ns         28.8 ns     25649394 bytes_per_second=530.626Gi/s items_per_second=34.7751M/s
// Allocate/32768                   28.8 ns         28.8 ns     24556833 bytes_per_second=1.03401Ti/s items_per_second=34.6955M/s
// Allocate/65536                   27.6 ns         27.6 ns     24834667 bytes_per_second=2.1602Ti/s items_per_second=36.2421M/s
// Allocate/131072                  27.7 ns         27.7 ns     24017700 bytes_per_second=4.30635Ti/s items_per_second=36.1243M/s
// Allocate/262144                  27.2 ns         27.2 ns     23559309 bytes_per_second=8.75476Ti/s items_per_second=36.7201M/s
// Allocate/524288                  26.9 ns         26.9 ns     25799785 bytes_per_second=17.6965Ti/s items_per_second=37.1123M/s
// Allocate/1048576                 27.1 ns         27.1 ns     25119944 bytes_per_second=35.1456Ti/s items_per_second=36.8528M/s

// IntAssigment
//   8.83 │ 60:   mov      %r14,%rdi
//  17.85 │     → call     operator new[](unsigned long)@plt
//  16.80 │       mov      %rax,-0x60(%rbp)
//  18.08 │       mov      %rax,%rdi
//  16.19 │     → call     operator delete[](void*)@plt
//   7.71 │       test     %r12,%r12
//   0.44 │     ↓ jle      1d1
//   5.45 │       dec      %r12
//   8.65 │     ↑ jne      60


// Note1: Both g++ and clang++ give the same result which call malloc and free internally.
// Note2: The time require to allocate increase from 10ns to 30ns at 1024B -> 2048B
BENCHMARK_MAIN();