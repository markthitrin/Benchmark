#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = (char*)calloc(array_size,sizeof(char));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    short* p = (short*)calloc(array_size,sizeof(int));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    int* p = (int*)calloc(array_size,sizeof(int));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long long* p = (long long*)calloc(array_size,sizeof(long long));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    float* p = (float*)calloc(array_size,sizeof(float));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    double* p = (double*)calloc(array_size,sizeof(double));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long double* p = (long double*)calloc(array_size,sizeof(long double));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
}

static void Allocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = (char*)calloc(array_size,sizeof(char));
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

// BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<20); 
// BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
BENCHMARK(Allocate)->RangeMultiplier(2)->Range(1,1<<30);

// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   13.3 ns         13.3 ns     52917191 bytes_per_second=71.9054Mi/s items_per_second=75.3983M/s
// CharAllocate/4                   13.2 ns         13.2 ns     52873451 bytes_per_second=288.136Mi/s items_per_second=75.5332M/s
// CharAllocate/16                  13.3 ns         13.3 ns     52462998 bytes_per_second=1.12422Gi/s items_per_second=75.4454M/s
// CharAllocate/64                  13.7 ns         13.7 ns     51039549 bytes_per_second=4.3407Gi/s items_per_second=72.8248M/s
// CharAllocate/256                 31.0 ns         31.0 ns     22593500 bytes_per_second=7.69209Gi/s items_per_second=32.263M/s
// CharAllocate/1024                31.5 ns         31.5 ns     22482595 bytes_per_second=30.3066Gi/s items_per_second=31.7788M/s
// CharAllocate/4096                51.3 ns         51.3 ns     13567659 bytes_per_second=74.3785Gi/s items_per_second=19.4979M/s
// CharAllocate/16384                154 ns          154 ns      4549585 bytes_per_second=98.9789Gi/s items_per_second=6.48668M/s
// CharAllocate/65536                547 ns          547 ns      1277484 bytes_per_second=111.508Gi/s items_per_second=1.82696M/s
// CharAllocate/262144              2108 ns         2108 ns       332820 bytes_per_second=115.836Gi/s items_per_second=474.463k/s
// CharAllocate/1048576            11972 ns        11970 ns        56551 bytes_per_second=81.5827Gi/s items_per_second=83.5407k/s

// ShortAllocate/1                  13.2 ns         13.2 ns     52864934 bytes_per_second=144.069Mi/s items_per_second=75.5337M/s
// ShortAllocate/4                  13.3 ns         13.3 ns     52810294 bytes_per_second=574.864Mi/s items_per_second=75.3485M/s
// ShortAllocate/16                 13.8 ns         13.8 ns     50866727 bytes_per_second=2.16411Gi/s items_per_second=72.6155M/s
// ShortAllocate/64                 30.8 ns         30.8 ns     22695100 bytes_per_second=3.87162Gi/s items_per_second=32.4775M/s
// ShortAllocate/256                31.7 ns         31.7 ns     22401817 bytes_per_second=15.0477Gi/s items_per_second=31.5574M/s
// ShortAllocate/1024               51.1 ns         51.1 ns     13901393 bytes_per_second=37.3606Gi/s items_per_second=19.5877M/s
// ShortAllocate/4096                156 ns          156 ns      4558106 bytes_per_second=48.8136Gi/s items_per_second=6.3981M/s
// ShortAllocate/16384               544 ns          544 ns      1277148 bytes_per_second=56.139Gi/s items_per_second=1.83956M/s
// ShortAllocate/65536              2100 ns         2099 ns       332689 bytes_per_second=58.1497Gi/s items_per_second=476.362k/s
// ShortAllocate/262144            11936 ns        11934 ns        56724 bytes_per_second=40.9158Gi/s items_per_second=83.7956k/s
// ShortAllocate/1048576          258146 ns       258085 ns         2744 bytes_per_second=7.56776Gi/s items_per_second=3.87469k/s

// IntAllocate/1                    13.3 ns         13.3 ns     52857132 bytes_per_second=287.829Mi/s items_per_second=75.4527M/s
// IntAllocate/4                    13.3 ns         13.3 ns     52773230 bytes_per_second=1.1189Gi/s items_per_second=75.0879M/s
// IntAllocate/16                   13.8 ns         13.8 ns     50821969 bytes_per_second=4.32545Gi/s items_per_second=72.569M/s
// IntAllocate/64                   31.0 ns         30.9 ns     22777428 bytes_per_second=7.70404Gi/s items_per_second=32.3131M/s
// IntAllocate/256                  31.7 ns         31.7 ns     22321376 bytes_per_second=30.1309Gi/s items_per_second=31.5946M/s
// IntAllocate/1024                 50.2 ns         50.2 ns     13959286 bytes_per_second=76.0637Gi/s items_per_second=19.9396M/s
// IntAllocate/4096                  155 ns          155 ns      4602726 bytes_per_second=98.397Gi/s items_per_second=6.44855M/s
// IntAllocate/16384                 548 ns          548 ns      1271528 bytes_per_second=111.306Gi/s items_per_second=1.82363M/s
// IntAllocate/65536                2101 ns         2101 ns       332605 bytes_per_second=116.199Gi/s items_per_second=475.95k/s
// IntAllocate/262144              12148 ns        12146 ns        59093 bytes_per_second=80.4033Gi/s items_per_second=82.3329k/s
// IntAllocate/1048576            256491 ns       256390 ns         2819 bytes_per_second=15.2356Gi/s items_per_second=3.9003k/s

// LongLongAllocate/1               13.3 ns         13.3 ns     52915540 bytes_per_second=575.317Mi/s items_per_second=75.4079M/s
// LongLongAllocate/4               13.3 ns         13.3 ns     52514704 bytes_per_second=2.23623Gi/s items_per_second=75.0353M/s
// LongLongAllocate/16              31.2 ns         31.2 ns     22456670 bytes_per_second=3.82231Gi/s items_per_second=32.0639M/s
// LongLongAllocate/64              31.8 ns         31.8 ns     21992219 bytes_per_second=15.0039Gi/s items_per_second=31.4656M/s
// LongLongAllocate/256             35.4 ns         35.4 ns     19517286 bytes_per_second=53.8306Gi/s items_per_second=28.2227M/s
// LongLongAllocate/1024            81.1 ns         81.1 ns      8725675 bytes_per_second=94.0726Gi/s items_per_second=12.3303M/s
// LongLongAllocate/4096             282 ns          282 ns      2475403 bytes_per_second=108.227Gi/s items_per_second=3.54637M/s
// LongLongAllocate/16384           1061 ns         1061 ns       657485 bytes_per_second=115.073Gi/s items_per_second=942.681k/s
// LongLongAllocate/65536           5016 ns         5015 ns       134698 bytes_per_second=97.3697Gi/s items_per_second=199.413k/s
// LongLongAllocate/262144         25844 ns        25841 ns        27544 bytes_per_second=75.5835Gi/s items_per_second=38.6987k/s
// LongLongAllocate/1048576      1548837 ns      1548128 ns          441 bytes_per_second=5.04642Gi/s items_per_second=645.942/s

// FloatAllocate/1                  13.5 ns         13.4 ns     52855812 bytes_per_second=283.666Mi/s items_per_second=74.3614M/s
// FloatAllocate/4                  13.3 ns         13.3 ns     52724659 bytes_per_second=1.12253Gi/s items_per_second=75.3315M/s
// FloatAllocate/16                 13.9 ns         13.9 ns     50932413 bytes_per_second=4.30273Gi/s items_per_second=72.1878M/s
// FloatAllocate/64                 31.1 ns         31.1 ns     22908199 bytes_per_second=7.67648Gi/s items_per_second=32.1975M/s
// FloatAllocate/256                31.3 ns         31.3 ns     22193311 bytes_per_second=30.5068Gi/s items_per_second=31.9887M/s
// FloatAllocate/1024               50.3 ns         50.3 ns     14121061 bytes_per_second=75.8044Gi/s items_per_second=19.8717M/s
// FloatAllocate/4096                153 ns          153 ns      4628010 bytes_per_second=99.6562Gi/s items_per_second=6.53107M/s
// FloatAllocate/16384               550 ns          550 ns      1278914 bytes_per_second=111.04Gi/s items_per_second=1.81928M/s
// FloatAllocate/65536              2117 ns         2116 ns       331320 bytes_per_second=115.362Gi/s items_per_second=472.521k/s
// FloatAllocate/262144            12137 ns        12136 ns        55998 bytes_per_second=80.4669Gi/s items_per_second=82.3981k/s
// FloatAllocate/1048576          254214 ns       254185 ns         2616 bytes_per_second=15.3677Gi/s items_per_second=3.93414k/s

// DoubleAllocate/1                 13.3 ns         13.3 ns     52747482 bytes_per_second=574.4Mi/s items_per_second=75.2877M/s
// DoubleAllocate/4                 13.4 ns         13.4 ns     52603958 bytes_per_second=2.23159Gi/s items_per_second=74.8797M/s
// DoubleAllocate/16                31.2 ns         31.2 ns     22789693 bytes_per_second=3.8256Gi/s items_per_second=32.0915M/s
// DoubleAllocate/64                31.8 ns         31.8 ns     22111361 bytes_per_second=14.9945Gi/s items_per_second=31.4458M/s
// DoubleAllocate/256               34.9 ns         34.9 ns     20345659 bytes_per_second=54.6729Gi/s items_per_second=28.6643M/s
// DoubleAllocate/1024              80.1 ns         80.1 ns      8592192 bytes_per_second=95.2733Gi/s items_per_second=12.4877M/s
// DoubleAllocate/4096               284 ns          284 ns      2485904 bytes_per_second=107.476Gi/s items_per_second=3.52176M/s
// DoubleAllocate/16384             1083 ns         1083 ns       645531 bytes_per_second=112.748Gi/s items_per_second=923.635k/s
// DoubleAllocate/65536             5102 ns         5102 ns       124285 bytes_per_second=95.7053Gi/s items_per_second=196.004k/s
// DoubleAllocate/262144           25470 ns        25464 ns        27786 bytes_per_second=76.7004Gi/s items_per_second=39.2706k/s
// DoubleAllocate/1048576        1534943 ns      1534627 ns          452 bytes_per_second=5.09081Gi/s items_per_second=651.624/s

// LongDoubleAllocate/1             13.3 ns         13.3 ns     52781291 bytes_per_second=1.12048Gi/s items_per_second=75.1943M/s
// LongDoubleAllocate/4             13.8 ns         13.8 ns     50532314 bytes_per_second=4.31052Gi/s items_per_second=72.3185M/s
// LongDoubleAllocate/16            32.5 ns         32.5 ns     21943376 bytes_per_second=7.33801Gi/s items_per_second=30.7778M/s
// LongDoubleAllocate/64            33.0 ns         33.0 ns     21601855 bytes_per_second=28.8823Gi/s items_per_second=30.2853M/s
// LongDoubleAllocate/256           51.8 ns         51.8 ns     13848140 bytes_per_second=73.594Gi/s items_per_second=19.2922M/s
// LongDoubleAllocate/1024           154 ns          154 ns      4549968 bytes_per_second=98.845Gi/s items_per_second=6.4779M/s
// LongDoubleAllocate/4096           553 ns          553 ns      1261457 bytes_per_second=110.32Gi/s items_per_second=1.80748M/s
// LongDoubleAllocate/16384         2115 ns         2114 ns       331391 bytes_per_second=115.464Gi/s items_per_second=472.939k/s
// LongDoubleAllocate/65536        12519 ns        12517 ns        56163 bytes_per_second=78.0177Gi/s items_per_second=79.8901k/s
// LongDoubleAllocate/262144      244430 ns       244366 ns         2620 bytes_per_second=15.9853Gi/s items_per_second=4.09222k/s
// LongDoubleAllocate/1048576    3011919 ns      3011109 ns          221 bytes_per_second=5.18912Gi/s items_per_second=332.104/s

// Allocate/1                13.2 ns         13.2 ns     51107693 bytes_per_second=72.4902Mi/s items_per_second=76.0115M/s
// Allocate/2                13.1 ns         13.1 ns     53046511 bytes_per_second=145.177Mi/s items_per_second=76.1143M/s
// Allocate/4                13.2 ns         13.2 ns     52971877 bytes_per_second=289.445Mi/s items_per_second=75.8764M/s
// Allocate/8                13.1 ns         13.1 ns     53112601 bytes_per_second=580.602Mi/s items_per_second=76.1007M/s
// Allocate/16               13.2 ns         13.2 ns     53019397 bytes_per_second=1.13241Gi/s items_per_second=75.995M/s
// Allocate/32               13.2 ns         13.2 ns     52731758 bytes_per_second=2.25575Gi/s items_per_second=75.6905M/s
// Allocate/64               13.6 ns         13.6 ns     51195763 bytes_per_second=4.37917Gi/s items_per_second=73.4703M/s
// Allocate/128              28.6 ns         28.6 ns     24377765 bytes_per_second=4.16168Gi/s items_per_second=34.9107M/s
// Allocate/256              29.1 ns         29.1 ns     23659898 bytes_per_second=8.20329Gi/s items_per_second=34.4071M/s
// Allocate/512              29.9 ns         29.9 ns     23493476 bytes_per_second=15.9529Gi/s items_per_second=33.4557M/s
// Allocate/1024             28.8 ns         28.8 ns     24198421 bytes_per_second=33.1116Gi/s items_per_second=34.72M/s
// Allocate/2048             33.3 ns         33.3 ns     21133741 bytes_per_second=57.3282Gi/s items_per_second=30.0565M/s
// Allocate/4096             48.4 ns         48.4 ns     14475220 bytes_per_second=78.8621Gi/s items_per_second=20.6732M/s
// Allocate/8192             79.1 ns         79.1 ns      8750300 bytes_per_second=96.5055Gi/s items_per_second=12.6492M/s
// Allocate/16384             150 ns          150 ns      4650017 bytes_per_second=101.656Gi/s items_per_second=6.66212M/s
// Allocate/32768             281 ns          281 ns      2492744 bytes_per_second=108.501Gi/s items_per_second=3.55535M/s
// Allocate/65536             526 ns          526 ns      1321206 bytes_per_second=116.076Gi/s items_per_second=1.90179M/s
// Allocate/131072           1021 ns         1021 ns       684736 bytes_per_second=119.523Gi/s items_per_second=979.133k/s
// Allocate/262144           2015 ns         2014 ns       347556 bytes_per_second=121.205Gi/s items_per_second=496.455k/s
// Allocate/524288           5408 ns         5408 ns       134886 bytes_per_second=90.2965Gi/s items_per_second=184.927k/s
// Allocate/1048576         11980 ns        11978 ns        58960 bytes_per_second=81.5274Gi/s items_per_second=83.4841k/s
// Allocate/2097152         25203 ns        25199 ns        27573 bytes_per_second=77.5091Gi/s items_per_second=39.6847k/s
// Allocate/4194304        207693 ns       207631 ns         3261 bytes_per_second=18.8135Gi/s items_per_second=4.81625k/s
// Allocate/8388608       1482144 ns      1481587 ns          456 bytes_per_second=5.27306Gi/s items_per_second=674.952/s
// Allocate/16777216      2998583 ns      2997760 ns          231 bytes_per_second=5.21222Gi/s items_per_second=333.582/s
// Allocate/33554432        11895 ns        11892 ns        59015 bytes_per_second=2.56622Ti/s items_per_second=84.09k/s
// Allocate/67108864        13247 ns        13224 ns        52642 bytes_per_second=4.6156Ti/s items_per_second=75.622k/s
// Allocate/134217728       14231 ns        14227 ns        48975 bytes_per_second=8.58012Ti/s items_per_second=70.2884k/s
// Allocate/268435456       15866 ns        15731 ns        43943 bytes_per_second=15.5199Ti/s items_per_second=63.5695k/s
// Allocate/536870912       19388 ns        18721 ns        37466 bytes_per_second=26.0817Ti/s items_per_second=53.4154k/s
// Allocate/1073741824      26556 ns        25716 ns        27156 bytes_per_second=37.9748Ti/s items_per_second=38.8862k/s
// CharAllocate
//   9.04 │ 58:┌─→mov      %r12,%rdi                                                                                                                                         ▒
//  11.51 │    │  mov      $0x1,%esi                                                                                                                                         ▒
//  20.64 │    │→ call     calloc@plt                                                                                                                                        ▒
//  10.15 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  18.19 │    │→ call     free@plt                                                                                                                                          ▒
//   8.70 │    │  test     %r13,%r13                                                                                                                                         ▒
//   4.24 │    │↓ jle      1bf                                                                                                                                               ▒
//   7.46 │    │  sub      $0x1,%r13                                                                                                                                         ▒
//  10.06 │    └──jne      58   
// IntAllocate
//   9.57 │ 58:┌─→mov      %r12,%rdi                                                                                                                                         ▒
//  12.22 │    │  mov      $0x4,%esi                                                                                                                                         ▒
//  20.60 │    │→ call     calloc@plt                                                                                                                                        ▒
//  11.17 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  18.94 │    │→ call     free@plt                                                                                                                                          ▒
//   9.84 │    │  test     %r13,%r13                                                                                                                                         ▒
//   1.45 │    │↓ jle      1c7                                                                                                                                               ▒
//   6.84 │    │  sub      $0x1,%r13                                                                                                                                         ▒
//   9.32 │    └──jne      58 



// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   13.1 ns         13.1 ns     53375937 bytes_per_second=72.6534Mi/s items_per_second=76.1827M/s
// CharAllocate/4                   13.1 ns         13.1 ns     53303342 bytes_per_second=290.77Mi/s items_per_second=76.2237M/s
// CharAllocate/16                  13.1 ns         13.1 ns     53256287 bytes_per_second=1.13546Gi/s items_per_second=76.1993M/s
// CharAllocate/64                  13.6 ns         13.6 ns     51364045 bytes_per_second=4.37835Gi/s items_per_second=73.4566M/s
// CharAllocate/256                 30.7 ns         30.7 ns     22933506 bytes_per_second=7.76183Gi/s items_per_second=32.5555M/s
// CharAllocate/1024                31.0 ns         31.0 ns     22569058 bytes_per_second=30.7749Gi/s items_per_second=32.2698M/s
// CharAllocate/4096                50.6 ns         50.6 ns     13811391 bytes_per_second=75.4623Gi/s items_per_second=19.782M/s
// CharAllocate/16384                151 ns          151 ns      4589012 bytes_per_second=100.919Gi/s items_per_second=6.61381M/s
// CharAllocate/65536                541 ns          541 ns      1267234 bytes_per_second=112.771Gi/s items_per_second=1.84763M/s
// CharAllocate/262144              2078 ns         2077 ns       337037 bytes_per_second=117.532Gi/s items_per_second=481.41k/s
// CharAllocate/1048576            11767 ns        11764 ns        58246 bytes_per_second=83.0128Gi/s items_per_second=85.0051k/s

// ShortAllocate/1                  13.1 ns         13.1 ns     53389702 bytes_per_second=145.398Mi/s items_per_second=76.2303M/s
// ShortAllocate/4                  13.1 ns         13.1 ns     53335853 bytes_per_second=581.627Mi/s items_per_second=76.235M/s
// ShortAllocate/16                 13.8 ns         13.8 ns     51455551 bytes_per_second=2.15972Gi/s items_per_second=72.4683M/s
// ShortAllocate/64                 30.3 ns         30.3 ns     23111794 bytes_per_second=3.93454Gi/s items_per_second=33.0053M/s
// ShortAllocate/256                31.0 ns         30.9 ns     22564159 bytes_per_second=15.4075Gi/s items_per_second=32.3119M/s
// ShortAllocate/1024               49.6 ns         49.6 ns     13959375 bytes_per_second=38.4425Gi/s items_per_second=20.1549M/s
// ShortAllocate/4096                150 ns          150 ns      4660939 bytes_per_second=50.7983Gi/s items_per_second=6.65824M/s
// ShortAllocate/16384               542 ns          542 ns      1292882 bytes_per_second=56.3463Gi/s items_per_second=1.84636M/s
// ShortAllocate/65536              2076 ns         2075 ns       337344 bytes_per_second=58.815Gi/s items_per_second=481.813k/s
// ShortAllocate/262144            11773 ns        11772 ns        59417 bytes_per_second=41.4799Gi/s items_per_second=84.9509k/s
// ShortAllocate/1048576          227062 ns       226999 ns         3076 bytes_per_second=8.60411Gi/s items_per_second=4.40531k/s

// IntAllocate/1                    13.1 ns         13.1 ns     50903377 bytes_per_second=290.963Mi/s items_per_second=76.2742M/s
// IntAllocate/4                    13.1 ns         13.1 ns     53345853 bytes_per_second=1.13675Gi/s items_per_second=76.2857M/s
// IntAllocate/16                   13.6 ns         13.6 ns     51398890 bytes_per_second=4.37533Gi/s items_per_second=73.4059M/s
// IntAllocate/64                   30.6 ns         30.6 ns     21814403 bytes_per_second=7.79448Gi/s items_per_second=32.6924M/s
// IntAllocate/256                  30.9 ns         30.9 ns     22775136 bytes_per_second=30.8621Gi/s items_per_second=32.3612M/s
// IntAllocate/1024                 49.8 ns         49.8 ns     14068333 bytes_per_second=76.6066Gi/s items_per_second=20.082M/s
// IntAllocate/4096                  150 ns          150 ns      4661532 bytes_per_second=101.629Gi/s items_per_second=6.66033M/s
// IntAllocate/16384                 539 ns          539 ns      1297032 bytes_per_second=113.178Gi/s items_per_second=1.85431M/s
// IntAllocate/65536                2075 ns         2075 ns       337394 bytes_per_second=117.656Gi/s items_per_second=481.918k/s
// IntAllocate/262144              11800 ns        11798 ns        59518 bytes_per_second=82.7722Gi/s items_per_second=84.7588k/s
// IntAllocate/1048576            222431 ns       222377 ns         3212 bytes_per_second=17.5659Gi/s items_per_second=4.49686k/s

// LongLongAllocate/1               13.1 ns         13.1 ns     53467696 bytes_per_second=582.561Mi/s items_per_second=76.3574M/s
// LongLongAllocate/4               13.2 ns         13.2 ns     53143930 bytes_per_second=2.26135Gi/s items_per_second=75.8784M/s
// LongLongAllocate/16              30.8 ns         30.8 ns     22731894 bytes_per_second=3.87318Gi/s items_per_second=32.4906M/s
// LongLongAllocate/64              32.0 ns         32.0 ns     21801505 bytes_per_second=14.9221Gi/s items_per_second=31.294M/s
// LongLongAllocate/256             33.9 ns         33.9 ns     20629596 bytes_per_second=56.3152Gi/s items_per_second=29.5254M/s
// LongLongAllocate/1024            79.7 ns         79.7 ns      8733326 bytes_per_second=95.7474Gi/s items_per_second=12.5498M/s
// LongLongAllocate/4096             279 ns          279 ns      2500762 bytes_per_second=109.244Gi/s items_per_second=3.57971M/s
// LongLongAllocate/16384           1054 ns         1054 ns       664234 bytes_per_second=115.822Gi/s items_per_second=948.815k/s
// LongLongAllocate/65536           4886 ns         4885 ns       143398 bytes_per_second=99.9542Gi/s items_per_second=204.706k/s
// LongLongAllocate/262144         24687 ns        24684 ns        28323 bytes_per_second=79.1255Gi/s items_per_second=40.5122k/s
// LongLongAllocate/1048576      1469048 ns      1468600 ns          432 bytes_per_second=5.31969Gi/s items_per_second=680.92/s

// FloatAllocate/1                  13.1 ns         13.1 ns     53462328 bytes_per_second=291.344Mi/s items_per_second=76.3742M/s
// FloatAllocate/4                  13.1 ns         13.1 ns     53352746 bytes_per_second=1.13737Gi/s items_per_second=76.3276M/s
// FloatAllocate/16                 13.6 ns         13.6 ns     51433355 bytes_per_second=4.37796Gi/s items_per_second=73.45M/s
// FloatAllocate/64                 30.7 ns         30.7 ns     22904951 bytes_per_second=7.77642Gi/s items_per_second=32.6167M/s
// FloatAllocate/256                30.8 ns         30.8 ns     22716859 bytes_per_second=30.9168Gi/s items_per_second=32.4186M/s
// FloatAllocate/1024               49.7 ns         49.7 ns     14087716 bytes_per_second=76.7652Gi/s items_per_second=20.1235M/s
// FloatAllocate/4096                150 ns          150 ns      4651849 bytes_per_second=101.516Gi/s items_per_second=6.65298M/s
// FloatAllocate/16384               541 ns          540 ns      1290310 bytes_per_second=112.939Gi/s items_per_second=1.8504M/s
// FloatAllocate/65536              2082 ns         2081 ns       336442 bytes_per_second=117.301Gi/s items_per_second=480.464k/s
// FloatAllocate/262144            11774 ns        11772 ns        56974 bytes_per_second=82.9583Gi/s items_per_second=84.9493k/s
// FloatAllocate/1048576          217683 ns       217619 ns         3121 bytes_per_second=17.95Gi/s items_per_second=4.59519k/s

// DoubleAllocate/1                 13.1 ns         13.1 ns     53449523 bytes_per_second=582.552Mi/s items_per_second=76.3563M/s
// DoubleAllocate/4                 13.2 ns         13.2 ns     46584371 bytes_per_second=2.26398Gi/s items_per_second=75.9667M/s
// DoubleAllocate/16                31.1 ns         31.1 ns     22485613 bytes_per_second=3.83269Gi/s items_per_second=32.1509M/s
// DoubleAllocate/64                31.9 ns         31.9 ns     21772087 bytes_per_second=14.944Gi/s items_per_second=31.3398M/s
// DoubleAllocate/256               34.0 ns         34.0 ns     20571103 bytes_per_second=56.082Gi/s items_per_second=29.4031M/s
// DoubleAllocate/1024              80.1 ns         80.0 ns      8611678 bytes_per_second=95.3214Gi/s items_per_second=12.494M/s
// DoubleAllocate/4096               279 ns          279 ns      2497978 bytes_per_second=109.235Gi/s items_per_second=3.57943M/s
// DoubleAllocate/16384             1053 ns         1053 ns       663413 bytes_per_second=115.959Gi/s items_per_second=949.936k/s
// DoubleAllocate/65536             4879 ns         4879 ns       143376 bytes_per_second=100.082Gi/s items_per_second=204.969k/s
// DoubleAllocate/262144           24806 ns        24802 ns        28384 bytes_per_second=78.7499Gi/s items_per_second=40.32k/s
// DoubleAllocate/1048576        1491090 ns      1490741 ns          453 bytes_per_second=5.24068Gi/s items_per_second=670.807/s

// LongDoubleAllocate/1             13.1 ns         13.1 ns     52667802 bytes_per_second=1.13456Gi/s items_per_second=76.1392M/s
// LongDoubleAllocate/4             13.6 ns         13.6 ns     50109965 bytes_per_second=4.37166Gi/s items_per_second=73.3443M/s
// LongDoubleAllocate/16            30.9 ns         30.9 ns     22741661 bytes_per_second=7.72054Gi/s items_per_second=32.3823M/s
// LongDoubleAllocate/64            31.0 ns         31.0 ns     22481906 bytes_per_second=30.7661Gi/s items_per_second=32.2606M/s
// LongDoubleAllocate/256           49.8 ns         49.8 ns     13826795 bytes_per_second=76.5804Gi/s items_per_second=20.0751M/s
// LongDoubleAllocate/1024           150 ns          150 ns      4655742 bytes_per_second=101.576Gi/s items_per_second=6.65689M/s
// LongDoubleAllocate/4096           541 ns          541 ns      1289815 bytes_per_second=112.772Gi/s items_per_second=1.84766M/s
// LongDoubleAllocate/16384         2082 ns         2081 ns       336343 bytes_per_second=117.309Gi/s items_per_second=480.499k/s
// LongDoubleAllocate/65536        11783 ns        11781 ns        59435 bytes_per_second=82.8928Gi/s items_per_second=84.8823k/s
// LongDoubleAllocate/262144      220562 ns       220505 ns         3207 bytes_per_second=17.715Gi/s items_per_second=4.53504k/s
// LongDoubleAllocate/1048576    2964577 ns      2963782 ns          238 bytes_per_second=5.27198Gi/s items_per_second=337.407/s

// Allocate/1                13.2 ns         13.2 ns     52741537 bytes_per_second=72.1365Mi/s items_per_second=75.6406M/s
// Allocate/2                13.2 ns         13.2 ns     52798342 bytes_per_second=144.298Mi/s items_per_second=75.6538M/s
// Allocate/4                13.2 ns         13.2 ns     53020048 bytes_per_second=289.186Mi/s items_per_second=75.8084M/s
// Allocate/8                13.2 ns         13.2 ns     53049193 bytes_per_second=577.65Mi/s items_per_second=75.7137M/s
// Allocate/16               13.2 ns         13.2 ns     52750170 bytes_per_second=1.12914Gi/s items_per_second=75.7756M/s
// Allocate/32               13.3 ns         13.3 ns     52514976 bytes_per_second=2.23908Gi/s items_per_second=75.1311M/s
// Allocate/64               13.7 ns         13.7 ns     50956334 bytes_per_second=4.34085Gi/s items_per_second=72.8274M/s
// Allocate/128              29.2 ns         29.2 ns     23439774 bytes_per_second=4.08776Gi/s items_per_second=34.2906M/s
// Allocate/256              28.9 ns         28.9 ns     24311258 bytes_per_second=8.26385Gi/s items_per_second=34.6611M/s
// Allocate/512              30.4 ns         30.4 ns     23286014 bytes_per_second=15.6822Gi/s items_per_second=32.888M/s
// Allocate/1024             29.6 ns         29.6 ns     23614142 bytes_per_second=32.1757Gi/s items_per_second=33.7387M/s
// Allocate/2048             33.4 ns         33.4 ns     21007359 bytes_per_second=57.0509Gi/s items_per_second=29.9111M/s
// Allocate/4096             48.9 ns         48.9 ns     14216944 bytes_per_second=78.0051Gi/s items_per_second=20.4486M/s
// Allocate/8192             79.5 ns         79.5 ns      8725209 bytes_per_second=95.9573Gi/s items_per_second=12.5773M/s
// Allocate/16384             151 ns          151 ns      4616686 bytes_per_second=100.857Gi/s items_per_second=6.60974M/s
// Allocate/32768             281 ns          281 ns      2492165 bytes_per_second=108.586Gi/s items_per_second=3.55815M/s
// Allocate/65536             528 ns          527 ns      1310625 bytes_per_second=115.711Gi/s items_per_second=1.89581M/s
// Allocate/131072           1023 ns         1023 ns       681906 bytes_per_second=119.358Gi/s items_per_second=977.78k/s
// Allocate/262144           2075 ns         2075 ns       337377 bytes_per_second=117.657Gi/s items_per_second=481.922k/s
// Allocate/524288           4986 ns         4985 ns       137079 bytes_per_second=97.9507Gi/s items_per_second=200.603k/s
// Allocate/1048576         12283 ns        12282 ns        56879 bytes_per_second=79.5121Gi/s items_per_second=81.4204k/s
// Allocate/2097152         24674 ns        24670 ns        28044 bytes_per_second=79.1715Gi/s items_per_second=40.5358k/s
// Allocate/4194304        230394 ns       230335 ns         3055 bytes_per_second=16.959Gi/s items_per_second=4.3415k/s
// Allocate/8388608       1460440 ns      1459765 ns          478 bytes_per_second=5.35189Gi/s items_per_second=685.042/s
// Allocate/16777216      3010459 ns      3009157 ns          234 bytes_per_second=5.19248Gi/s items_per_second=332.319/s
// Allocate/33554432         9799 ns         9377 ns        74027 bytes_per_second=3.25464Ti/s items_per_second=106.648k/s
// Allocate/67108864        10227 ns         9789 ns        70774 bytes_per_second=6.23514Ti/s items_per_second=102.157k/s
// Allocate/134217728       12087 ns        11641 ns        59714 bytes_per_second=10.486Ti/s items_per_second=85.9016k/s
// Allocate/268435456       13894 ns        13443 ns        51884 bytes_per_second=18.1611Ti/s items_per_second=74.3877k/s
// Allocate/536870912       17410 ns        16940 ns        41301 bytes_per_second=28.8236Ti/s items_per_second=59.0307k/s
// Allocate/1073741824      24421 ns        23936 ns        29165 bytes_per_second=40.7985Ti/s items_per_second=41.7777k/s
// CharAllocate
//   7.92 │ 40:┌─→mov      $0x1,%esi                                                                                                                                         ▒
//   7.65 │    │  mov      %r14,%rdi                                                                                                                                         ◆
//  24.25 │    │→ call     calloc@plt                                                                                                                                        ▒
//   8.39 │    │  mov      %rax,-0x60(%rbp)                                                                                                                                  ▒
//   6.23 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  23.32 │    │→ call     free@plt                                                                                                                                          ▒
//   6.43 │    │  test     %r12,%r12                                                                                                                                         ▒
//   2.65 │    │↓ jle      1b2                                                                                                                                               ▒
//   5.16 │    │  dec      %r12                                                                                                                                              ▒
//   8.00 │    └──jne      40  
// IntAllocate
//   6.69 │ 40:┌─→mov      $0x4,%esi                                                                                                                                         ▒
//  10.60 │    │  mov      %r14,%rdi                                                                                                                                         ▒
//  24.89 │    │→ call     calloc@plt                                                                                                                                        ▒
//   6.69 │    │  mov      %rax,-0x60(%rbp)                                                                                                                                  ▒
//   5.00 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  17.97 │    │→ call     free@plt                                                                                                                                          ▒
//  11.15 │    │  test     %r12,%r12                                                                                                                                         ▒
//   0.93 │    │↓ jle      1b6                                                                                                                                               ▒
//   7.16 │    │  dec      %r12                                                                                                                                              ▒
//   8.92 │    └──jne      40

// Note1: Both g++ and clang++ just call calloc and the calloc use malloc with __memset_avx2)unaligned_erms
// Note2: Look liks around 32MB, the program use another ways to use otherway to allocate and set zero like __munmap.
// Note3: From ChatGPT :::
//        For large memory requests (often over 128 KB, though this threshold can vary), 

//        calloc in glibc typically does not use malloc in the traditional sense. Instead, 
//        it switches to memory-mapped pages by using the mmap system call with flags like MAP_ANONYMOUS 
//        and MAP_PRIVATE.

//        The mmap system call requests memory directly from the kernel, bypassing the memory pool 
//        usually managed by malloc. This allows calloc to access large contiguous memory blocks directly from the OS.

//        Zero-Initialization with mmap: When mmap is used with MAP_ANONYMOUS, 
//        the kernel automatically provides a zero-initialized memory region, effectively skipping the need 
//        for a memset call. This can make calloc for large allocations faster, as it avoids redundant zeroing operations.

BENCHMARK_MAIN();