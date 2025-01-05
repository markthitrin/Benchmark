#include <benchmark/benchmark.h>
#include <stdlib.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = (char*)malloc(sizeof(char)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    short* p = (short*)malloc(sizeof(int)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    int* p = (int*)malloc(sizeof(int)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long long* p = (long long*)malloc(sizeof(long long)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    float* p = (float*)malloc(sizeof(float)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    double* p = (double*)malloc(sizeof(double)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long double* p = (long double*)malloc(sizeof(long double)*array_size);
    escape(p);
    free(p);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
}

static void Allocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = (char*)malloc(sizeof(char)*array_size);;
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
// CharAllocate/1                   7.73 ns         7.72 ns     91087906 bytes_per_second=123.457Mi/s items_per_second=129.454M/s
// CharAllocate/4                   7.67 ns         7.67 ns     91089082 bytes_per_second=497.5Mi/s items_per_second=130.417M/s
// CharAllocate/16                  7.81 ns         7.80 ns     89623938 bytes_per_second=1.9095Gi/s items_per_second=128.144M/s
// CharAllocate/64                  7.81 ns         7.81 ns     89571157 bytes_per_second=7.63223Gi/s items_per_second=128.048M/s
// CharAllocate/256                 7.80 ns         7.80 ns     89486151 bytes_per_second=30.5613Gi/s items_per_second=128.183M/s
// CharAllocate/1024                7.81 ns         7.80 ns     89643313 bytes_per_second=122.203Gi/s items_per_second=128.139M/s
// CharAllocate/4096                23.5 ns         23.5 ns     31005787 bytes_per_second=162.105Gi/s items_per_second=42.4949M/s
// CharAllocate/16384               22.0 ns         22.0 ns     31597485 bytes_per_second=694.201Gi/s items_per_second=45.4952M/s
// CharAllocate/65536               22.2 ns         22.2 ns     31535240 bytes_per_second=2.6836Ti/s items_per_second=45.0233M/s
// CharAllocate/262144              22.4 ns         22.4 ns     31529985 bytes_per_second=10.626Ti/s items_per_second=44.5688M/s
// CharAllocate/1048576             22.2 ns         22.2 ns     32142022 bytes_per_second=42.9158Ti/s items_per_second=45.0005M/s

// ShortAllocate/1                  7.66 ns         7.66 ns     91148675 bytes_per_second=249.061Mi/s items_per_second=130.58M/s
// ShortAllocate/4                  7.81 ns         7.81 ns     89827374 bytes_per_second=976.985Mi/s items_per_second=128.055M/s
// ShortAllocate/16                 7.79 ns         7.79 ns     89679866 bytes_per_second=3.8242Gi/s items_per_second=128.319M/s
// ShortAllocate/64                 7.81 ns         7.80 ns     89632514 bytes_per_second=15.2757Gi/s items_per_second=128.142M/s
// ShortAllocate/256                7.79 ns         7.79 ns     89733999 bytes_per_second=61.2156Gi/s items_per_second=128.378M/s
// ShortAllocate/1024               23.9 ns         23.9 ns     31918127 bytes_per_second=79.8069Gi/s items_per_second=41.8418M/s
// ShortAllocate/4096               23.9 ns         23.9 ns     31882000 bytes_per_second=319.706Gi/s items_per_second=41.9046M/s
// ShortAllocate/16384              23.6 ns         23.6 ns     31370498 bytes_per_second=1.26255Ti/s items_per_second=42.364M/s
// ShortAllocate/65536              22.5 ns         22.5 ns     32271735 bytes_per_second=5.3022Ti/s items_per_second=44.4781M/s
// ShortAllocate/262144             22.1 ns         22.1 ns     32180714 bytes_per_second=21.5896Ti/s items_per_second=45.2766M/s
// ShortAllocate/1048576            21.7 ns         21.7 ns     32272942 bytes_per_second=87.8545Ti/s items_per_second=46.0611M/s

// IntAllocate/1                    7.73 ns         7.72 ns     91626023 bytes_per_second=493.915Mi/s items_per_second=129.477M/s
// IntAllocate/4                    7.80 ns         7.80 ns     89036569 bytes_per_second=1.91103Gi/s items_per_second=128.247M/s
// IntAllocate/16                   7.80 ns         7.80 ns     89773776 bytes_per_second=7.64596Gi/s items_per_second=128.278M/s
// IntAllocate/64                   7.92 ns         7.92 ns     89755785 bytes_per_second=30.1031Gi/s items_per_second=126.261M/s
// IntAllocate/256                  7.81 ns         7.81 ns     89899934 bytes_per_second=122.162Gi/s items_per_second=128.097M/s
// IntAllocate/1024                 22.0 ns         21.9 ns     31891092 bytes_per_second=173.803Gi/s items_per_second=45.5615M/s
// IntAllocate/4096                 22.0 ns         22.0 ns     31897051 bytes_per_second=695.125Gi/s items_per_second=45.5557M/s
// IntAllocate/16384                22.3 ns         22.3 ns     31535832 bytes_per_second=2.67635Ti/s items_per_second=44.9016M/s
// IntAllocate/65536                21.7 ns         21.7 ns     32258209 bytes_per_second=10.9861Ti/s items_per_second=46.0789M/s
// IntAllocate/262144               21.7 ns         21.7 ns     32252534 bytes_per_second=43.9514Ti/s items_per_second=46.0864M/s
// IntAllocate/1048576              21.7 ns         21.7 ns     32254900 bytes_per_second=175.632Ti/s items_per_second=46.0408M/s

// LongLongAllocate/1               7.73 ns         7.73 ns     90682687 bytes_per_second=987.166Mi/s items_per_second=129.39M/s
// LongLongAllocate/4               7.82 ns         7.82 ns     89890877 bytes_per_second=3.81321Gi/s items_per_second=127.95M/s
// LongLongAllocate/16              7.92 ns         7.92 ns     90002835 bytes_per_second=15.0553Gi/s items_per_second=126.293M/s
// LongLongAllocate/64              7.83 ns         7.83 ns     88171500 bytes_per_second=60.9255Gi/s items_per_second=127.77M/s
// LongLongAllocate/256             22.0 ns         21.9 ns     31902287 bytes_per_second=86.9077Gi/s items_per_second=45.5646M/s
// LongLongAllocate/1024            21.9 ns         21.9 ns     31603681 bytes_per_second=347.679Gi/s items_per_second=45.5709M/s
// LongLongAllocate/4096            22.4 ns         22.4 ns     31200177 bytes_per_second=1.32878Ti/s items_per_second=44.5865M/s
// LongLongAllocate/16384           22.7 ns         22.7 ns     30802601 bytes_per_second=5.25856Ti/s items_per_second=44.112M/s
// LongLongAllocate/65536           22.2 ns         22.2 ns     31537878 bytes_per_second=21.4865Ti/s items_per_second=45.0604M/s
// LongLongAllocate/262144          22.3 ns         22.3 ns     31543402 bytes_per_second=85.3959Ti/s items_per_second=44.772M/s
// LongLongAllocate/1048576         21.8 ns         21.8 ns     32261985 bytes_per_second=350.301Ti/s items_per_second=45.9146M/s

// FloatAllocate/1                  7.65 ns         7.64 ns     91214060 bytes_per_second=499.041Mi/s items_per_second=130.821M/s
// FloatAllocate/4                  7.80 ns         7.80 ns     89660638 bytes_per_second=1.91095Gi/s items_per_second=128.242M/s
// FloatAllocate/16                 7.80 ns         7.80 ns     89631192 bytes_per_second=7.64357Gi/s items_per_second=128.238M/s
// FloatAllocate/64                 7.79 ns         7.79 ns     89808185 bytes_per_second=30.5987Gi/s items_per_second=128.34M/s
// FloatAllocate/256                7.79 ns         7.79 ns     89599437 bytes_per_second=122.407Gi/s items_per_second=128.353M/s
// FloatAllocate/1024               22.1 ns         22.1 ns     29992581 bytes_per_second=172.65Gi/s items_per_second=45.2592M/s
// FloatAllocate/4096               21.9 ns         21.9 ns     31911140 bytes_per_second=695.418Gi/s items_per_second=45.5749M/s
// FloatAllocate/16384              22.2 ns         22.2 ns     31558909 bytes_per_second=2.68679Ti/s items_per_second=45.0769M/s
// FloatAllocate/65536              22.3 ns         22.3 ns     30874057 bytes_per_second=10.7056Ti/s items_per_second=44.9026M/s
// FloatAllocate/262144             21.7 ns         21.7 ns     32258730 bytes_per_second=43.9547Ti/s items_per_second=46.0899M/s
// FloatAllocate/1048576            21.7 ns         21.7 ns     32258235 bytes_per_second=175.82Ti/s items_per_second=46.0902M/s

// DoubleAllocate/1                 7.66 ns         7.66 ns     90587468 bytes_per_second=996.263Mi/s items_per_second=130.582M/s
// DoubleAllocate/4                 7.79 ns         7.79 ns     89598677 bytes_per_second=3.82464Gi/s items_per_second=128.334M/s
// DoubleAllocate/16                7.80 ns         7.80 ns     89647579 bytes_per_second=15.2926Gi/s items_per_second=128.284M/s
// DoubleAllocate/64                7.84 ns         7.83 ns     89181646 bytes_per_second=60.8693Gi/s items_per_second=127.652M/s
// DoubleAllocate/256               22.0 ns         21.9 ns     31888568 bytes_per_second=86.9078Gi/s items_per_second=45.5647M/s
// DoubleAllocate/1024              21.9 ns         21.9 ns     31888231 bytes_per_second=347.718Gi/s items_per_second=45.5762M/s
// DoubleAllocate/4096              22.5 ns         22.5 ns     31203375 bytes_per_second=1.32546Ti/s items_per_second=44.4749M/s
// DoubleAllocate/16384             22.7 ns         22.7 ns     30868452 bytes_per_second=5.25685Ti/s items_per_second=44.0977M/s
// DoubleAllocate/65536             22.2 ns         22.2 ns     31539904 bytes_per_second=21.4903Ti/s items_per_second=45.0685M/s
// DoubleAllocate/262144            22.2 ns         22.2 ns     31559029 bytes_per_second=85.9553Ti/s items_per_second=45.0653M/s
// DoubleAllocate/1048576           22.2 ns         22.2 ns     31554384 bytes_per_second=343.811Ti/s items_per_second=45.064M/s

// LongDoubleAllocate/1             9.04 ns         9.04 ns     77565387 bytes_per_second=1.64782Gi/s items_per_second=110.583M/s
// LongDoubleAllocate/4             9.03 ns         9.03 ns     77557129 bytes_per_second=6.60295Gi/s items_per_second=110.779M/s
// LongDoubleAllocate/16            9.02 ns         9.02 ns     77559523 bytes_per_second=26.4259Gi/s items_per_second=110.838M/s
// LongDoubleAllocate/64            9.02 ns         9.02 ns     77567445 bytes_per_second=105.716Gi/s items_per_second=110.851M/s
// LongDoubleAllocate/256           22.2 ns         22.2 ns     31910598 bytes_per_second=172.123Gi/s items_per_second=45.1211M/s
// LongDoubleAllocate/1024          21.9 ns         21.9 ns     31889161 bytes_per_second=695.333Gi/s items_per_second=45.5693M/s
// LongDoubleAllocate/4096          22.2 ns         22.2 ns     31542233 bytes_per_second=2.6854Ti/s items_per_second=45.0535M/s
// LongDoubleAllocate/16384         22.2 ns         22.2 ns     32072081 bytes_per_second=10.7574Ti/s items_per_second=45.1199M/s
// LongDoubleAllocate/65536         21.7 ns         21.7 ns     32183677 bytes_per_second=43.9247Ti/s items_per_second=46.0584M/s
// LongDoubleAllocate/262144        21.7 ns         21.7 ns     32228008 bytes_per_second=175.705Ti/s items_per_second=46.06M/s
// LongDoubleAllocate/1048576       22.2 ns         22.2 ns     31534578 bytes_per_second=687.26Ti/s items_per_second=45.0403M/s

// Allocate/1                7.57 ns         7.57 ns     90778236 bytes_per_second=125.955Mi/s items_per_second=132.073M/s
// Allocate/2                7.57 ns         7.57 ns     91686152 bytes_per_second=251.903Mi/s items_per_second=132.07M/s
// Allocate/4                7.56 ns         7.56 ns     91623658 bytes_per_second=504.826Mi/s items_per_second=132.337M/s
// Allocate/8                7.56 ns         7.56 ns     91315340 bytes_per_second=1009.17Mi/s items_per_second=132.274M/s
// Allocate/16               7.80 ns         7.80 ns     90041725 bytes_per_second=1.91097Gi/s items_per_second=128.243M/s
// Allocate/32               7.73 ns         7.73 ns     90267095 bytes_per_second=3.85735Gi/s items_per_second=129.431M/s
// Allocate/64               7.75 ns         7.75 ns     89923791 bytes_per_second=7.68921Gi/s items_per_second=129.004M/s
// Allocate/128              7.76 ns         7.76 ns     90038952 bytes_per_second=15.3626Gi/s items_per_second=128.871M/s
// Allocate/256              7.76 ns         7.76 ns     90073972 bytes_per_second=30.7302Gi/s items_per_second=128.892M/s
// Allocate/512              7.79 ns         7.79 ns     89566518 bytes_per_second=61.1952Gi/s items_per_second=128.336M/s
// Allocate/1024             7.76 ns         7.76 ns     89865257 bytes_per_second=122.966Gi/s items_per_second=128.939M/s
// Allocate/2048             19.0 ns         19.0 ns     36910973 bytes_per_second=100.497Gi/s items_per_second=52.6891M/s
// Allocate/4096             19.0 ns         19.0 ns     36994822 bytes_per_second=201.049Gi/s items_per_second=52.7038M/s
// Allocate/8192             18.9 ns         18.9 ns     36948000 bytes_per_second=402.764Gi/s items_per_second=52.7911M/s
// Allocate/16384            19.0 ns         19.0 ns     36940669 bytes_per_second=805.124Gi/s items_per_second=52.7646M/s
// Allocate/32768            22.2 ns         22.2 ns     31501202 bytes_per_second=1.3408Ti/s items_per_second=44.9897M/s
// Allocate/65536            22.5 ns         22.5 ns     31158475 bytes_per_second=2.65261Ti/s items_per_second=44.5034M/s
// Allocate/131072           22.5 ns         22.5 ns     31124123 bytes_per_second=5.29874Ti/s items_per_second=44.449M/s
// Allocate/262144           22.6 ns         22.6 ns     31807399 bytes_per_second=10.5366Ti/s items_per_second=44.1938M/s
// Allocate/524288           22.0 ns         22.0 ns     32397991 bytes_per_second=21.6911Ti/s items_per_second=45.4896M/s
// Allocate/1048576          22.1 ns         22.1 ns     32518378 bytes_per_second=43.0951Ti/s items_per_second=45.1885M/s
// Allocate/2097152          22.5 ns         22.5 ns     32221250 bytes_per_second=84.9044Ti/s items_per_second=44.5143M/s
// Allocate/4194304          22.2 ns         22.2 ns     31807033 bytes_per_second=172.091Ti/s items_per_second=45.1126M/s
// Allocate/8388608          21.9 ns         21.9 ns     31753980 bytes_per_second=347.711Ti/s items_per_second=45.5752M/s
// Allocate/16777216         21.9 ns         21.9 ns     31873095 bytes_per_second=695.304Ti/s items_per_second=45.5675M/s
// Allocate/33554432        11874 ns        11859 ns        59261 bytes_per_second=2.57333Ti/s items_per_second=84.3228k/s
// Allocate/67108864        12308 ns        12307 ns        56573 bytes_per_second=4.95957Ti/s items_per_second=81.2576k/s
// Allocate/134217728       13173 ns        13169 ns        52740 bytes_per_second=9.26928Ti/s items_per_second=75.9339k/s
// Allocate/268435456       14993 ns        14968 ns        46551 bytes_per_second=16.3111Ti/s items_per_second=66.8105k/s
// Allocate/536870912       18436 ns        18433 ns        38035 bytes_per_second=26.4896Ti/s items_per_second=54.2508k/s
// Allocate/1073741824      26297 ns        26273 ns        26438 bytes_per_second=37.1694Ti/s items_per_second=38.0615k/s
// CharAllocate
//  17.15 │ 70:┌─→mov      %r13,%rdi                                                                                                                                         ▒
//  27.38 │    │→ call     malloc@plt                                                                                                                                        ▒
//   9.88 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  21.00 │    │→ call     free@plt                                                                                                                                          ▒
//   9.25 │ 80:│  sub      $0x1,%r12                                                                                                                                         ▒
//  15.34 │    └──jne      70 
// ShortAllocate
//  17.31 │ 70:┌─→mov      %r13,%rdi                                                                                                                                         ▒
//  27.00 │    │→ call     malloc@plt                                                                                                                                        ▒
//  10.66 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  19.09 │    │→ call     free@plt                                                                                                                                          ▒
//  10.48 │ 80:│  sub      $0x1,%r12                                                                                                                                         ▒
//  15.45 │    └──jne      70

// -   96.03%     4.93%  MallocAllocate.  MallocAllocate.exe    [.] Allocate(benchmark::State&)                                                                              ▒
//    - 91.10% Allocate(benchmark::State&)                                                                                                                                   ▒
//       + 38.88% malloc                                                                                                                                                     ▒
//       + 36.36% cfree@GLIBC_2.2.5                                                                                                                                          ▒
//       + 11.77% __munmap                                                                                                                                                   ▒
//         0.60% _int_free_maybe_consolidate



// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// CharAllocate/1                   7.91 ns         7.91 ns     89317972 bytes_per_second=120.573Mi/s items_per_second=126.43M/s
// CharAllocate/4                   7.85 ns         7.85 ns     89037942 bytes_per_second=485.874Mi/s items_per_second=127.369M/s
// CharAllocate/16                  8.00 ns         8.00 ns     88266399 bytes_per_second=1.86378Gi/s items_per_second=125.076M/s
// CharAllocate/64                  7.93 ns         7.93 ns     88254537 bytes_per_second=7.51765Gi/s items_per_second=126.125M/s
// CharAllocate/256                 8.09 ns         8.09 ns     85783362 bytes_per_second=29.4638Gi/s items_per_second=123.58M/s
// CharAllocate/1024                7.93 ns         7.93 ns     87020357 bytes_per_second=120.284Gi/s items_per_second=126.127M/s
// CharAllocate/4096                22.3 ns         22.3 ns     28026191 bytes_per_second=170.825Gi/s items_per_second=44.7806M/s
// CharAllocate/16384               22.4 ns         22.4 ns     29932666 bytes_per_second=682.38Gi/s items_per_second=44.7204M/s
// CharAllocate/65536               22.6 ns         22.6 ns     30374920 bytes_per_second=2.63687Ti/s items_per_second=44.2394M/s
// CharAllocate/262144              22.1 ns         22.1 ns     31521910 bytes_per_second=10.7722Ti/s items_per_second=45.1817M/s
// CharAllocate/1048576             22.2 ns         22.2 ns     31625561 bytes_per_second=42.8913Ti/s items_per_second=44.9747M/s

// ShortAllocate/1                  7.86 ns         7.86 ns     88841710 bytes_per_second=242.683Mi/s items_per_second=127.236M/s
// ShortAllocate/4                  7.95 ns         7.94 ns     88121948 bytes_per_second=960.398Mi/s items_per_second=125.881M/s
// ShortAllocate/16                 7.95 ns         7.95 ns     88048845 bytes_per_second=3.74948Gi/s items_per_second=125.812M/s
// ShortAllocate/64                 7.95 ns         7.95 ns     85663456 bytes_per_second=14.998Gi/s items_per_second=125.812M/s
// ShortAllocate/256                7.95 ns         7.94 ns     88034045 bytes_per_second=60.0185Gi/s items_per_second=125.868M/s
// ShortAllocate/1024               23.7 ns         23.7 ns     30976489 bytes_per_second=80.3447Gi/s items_per_second=42.1237M/s
// ShortAllocate/4096               23.6 ns         23.6 ns     31300123 bytes_per_second=323.79Gi/s items_per_second=42.4399M/s
// ShortAllocate/16384              22.6 ns         22.6 ns     30991406 bytes_per_second=1.32002Ti/s items_per_second=44.2926M/s
// ShortAllocate/65536              22.1 ns         22.1 ns     31563585 bytes_per_second=5.38578Ti/s items_per_second=45.1792M/s
// ShortAllocate/262144             22.1 ns         22.1 ns     31631336 bytes_per_second=21.5343Ti/s items_per_second=45.1608M/s
// ShortAllocate/1048576            22.2 ns         22.2 ns     31617756 bytes_per_second=86.0769Ti/s items_per_second=45.1291M/s

// IntAllocate/1                    7.92 ns         7.92 ns     88879218 bytes_per_second=481.65Mi/s items_per_second=126.262M/s
// IntAllocate/4                    7.94 ns         7.94 ns     88025129 bytes_per_second=1.87767Gi/s items_per_second=126.009M/s
// IntAllocate/16                   7.95 ns         7.95 ns     87967822 bytes_per_second=7.49614Gi/s items_per_second=125.764M/s
// IntAllocate/64                   7.96 ns         7.96 ns     88018930 bytes_per_second=29.9634Gi/s items_per_second=125.675M/s
// IntAllocate/256                  7.97 ns         7.96 ns     88142013 bytes_per_second=119.736Gi/s items_per_second=125.552M/s
// IntAllocate/1024                 22.4 ns         22.4 ns     31273186 bytes_per_second=170.45Gi/s items_per_second=44.6825M/s
// IntAllocate/4096                 22.4 ns         22.4 ns     31284882 bytes_per_second=681.779Gi/s items_per_second=44.6811M/s
// IntAllocate/16384                22.6 ns         22.6 ns     30942023 bytes_per_second=2.63428Ti/s items_per_second=44.1959M/s
// IntAllocate/65536                22.2 ns         22.2 ns     31590173 bytes_per_second=10.755Ti/s items_per_second=45.1096M/s
// IntAllocate/262144               22.2 ns         22.2 ns     31599650 bytes_per_second=43.0147Ti/s items_per_second=45.1042M/s
// IntAllocate/1048576              22.2 ns         22.2 ns     31564766 bytes_per_second=171.798Ti/s items_per_second=45.0359M/s

// LongLongAllocate/1               7.87 ns         7.87 ns     87641687 bytes_per_second=969.196Mi/s items_per_second=127.035M/s
// LongLongAllocate/4               7.95 ns         7.94 ns     88178398 bytes_per_second=3.75145Gi/s items_per_second=125.878M/s
// LongLongAllocate/16              7.95 ns         7.95 ns     87933569 bytes_per_second=14.9961Gi/s items_per_second=125.797M/s
// LongLongAllocate/64              7.98 ns         7.98 ns     87902316 bytes_per_second=59.7799Gi/s items_per_second=125.368M/s
// LongLongAllocate/256             22.4 ns         22.4 ns     31243537 bytes_per_second=85.165Gi/s items_per_second=44.651M/s
// LongLongAllocate/1024            22.4 ns         22.4 ns     31266378 bytes_per_second=340.636Gi/s items_per_second=44.6478M/s
// LongLongAllocate/4096            22.9 ns         22.9 ns     30570670 bytes_per_second=1.30369Ti/s items_per_second=43.7444M/s
// LongLongAllocate/16384           23.1 ns         23.1 ns     30271078 bytes_per_second=5.15825Ti/s items_per_second=43.2705M/s
// LongLongAllocate/65536           22.7 ns         22.7 ns     30867766 bytes_per_second=21.0452Ti/s items_per_second=44.135M/s
// LongLongAllocate/262144          23.0 ns         23.0 ns     30890181 bytes_per_second=83.0949Ti/s items_per_second=43.5656M/s
// LongLongAllocate/1048576         22.2 ns         22.2 ns     31582834 bytes_per_second=344.144Ti/s items_per_second=45.1077M/s

// FloatAllocate/1                  7.89 ns         7.88 ns     88722682 bytes_per_second=483.829Mi/s items_per_second=126.833M/s
// FloatAllocate/4                  7.96 ns         7.96 ns     87799834 bytes_per_second=1.87279Gi/s items_per_second=125.681M/s
// FloatAllocate/16                 7.98 ns         7.98 ns     87729500 bytes_per_second=7.46968Gi/s items_per_second=125.32M/s
// FloatAllocate/64                 7.96 ns         7.96 ns     87794329 bytes_per_second=29.9674Gi/s items_per_second=125.692M/s
// FloatAllocate/256                7.95 ns         7.95 ns     84745275 bytes_per_second=119.905Gi/s items_per_second=125.73M/s
// FloatAllocate/1024               22.4 ns         22.4 ns     31227107 bytes_per_second=170.197Gi/s items_per_second=44.6162M/s
// FloatAllocate/4096               23.2 ns         23.2 ns     31218570 bytes_per_second=658.346Gi/s items_per_second=43.1454M/s
// FloatAllocate/16384              22.7 ns         22.6 ns     30901204 bytes_per_second=2.63166Ti/s items_per_second=44.1519M/s
// FloatAllocate/65536              22.2 ns         22.2 ns     31562414 bytes_per_second=10.7517Ti/s items_per_second=45.0958M/s
// FloatAllocate/262144             22.9 ns         22.9 ns     30816967 bytes_per_second=41.5923Ti/s items_per_second=43.6127M/s
// FloatAllocate/1048576            22.2 ns         22.2 ns     31572203 bytes_per_second=172.056Ti/s items_per_second=45.1034M/s

// DoubleAllocate/1                 9.30 ns         9.29 ns     81570215 bytes_per_second=820.918Mi/s items_per_second=107.599M/s
// DoubleAllocate/4                 7.96 ns         7.95 ns     87535369 bytes_per_second=3.7465Gi/s items_per_second=125.712M/s
// DoubleAllocate/16                9.44 ns         9.44 ns     75130377 bytes_per_second=12.6261Gi/s items_per_second=105.915M/s
// DoubleAllocate/64                9.03 ns         9.03 ns     89138856 bytes_per_second=52.8046Gi/s items_per_second=110.739M/s
// DoubleAllocate/256               22.8 ns         22.8 ns     31710006 bytes_per_second=83.4969Gi/s items_per_second=43.7764M/s
// DoubleAllocate/1024              23.8 ns         23.8 ns     30648102 bytes_per_second=321.026Gi/s items_per_second=42.0776M/s
// DoubleAllocate/4096              23.4 ns         23.4 ns     30993229 bytes_per_second=1.2739Ti/s items_per_second=42.7451M/s
// DoubleAllocate/16384             24.5 ns         24.5 ns     31250053 bytes_per_second=4.87355Ti/s items_per_second=40.8823M/s
// DoubleAllocate/65536             22.6 ns         22.6 ns     30906870 bytes_per_second=21.1198Ti/s items_per_second=44.2914M/s
// DoubleAllocate/262144            22.6 ns         22.6 ns     31892206 bytes_per_second=84.4585Ti/s items_per_second=44.2806M/s
// DoubleAllocate/1048576           22.0 ns         22.0 ns     31882771 bytes_per_second=346.557Ti/s items_per_second=45.424M/s

// LongDoubleAllocate/1             7.84 ns         7.84 ns     73139399 bytes_per_second=1.9008Gi/s items_per_second=127.56M/s
// LongDoubleAllocate/4             7.84 ns         7.83 ns     89297343 bytes_per_second=7.60858Gi/s items_per_second=127.651M/s
// LongDoubleAllocate/16            7.83 ns         7.83 ns     88851799 bytes_per_second=30.4524Gi/s items_per_second=127.727M/s
// LongDoubleAllocate/64            7.84 ns         7.84 ns     89067178 bytes_per_second=121.61Gi/s items_per_second=127.517M/s
// LongDoubleAllocate/256           22.3 ns         22.3 ns     31582476 bytes_per_second=171.249Gi/s items_per_second=44.8919M/s
// LongDoubleAllocate/1024          22.2 ns         22.2 ns     31416403 bytes_per_second=688.714Gi/s items_per_second=45.1356M/s
// LongDoubleAllocate/4096          22.4 ns         22.4 ns     31220591 bytes_per_second=2.65843Ti/s items_per_second=44.6011M/s
// LongDoubleAllocate/16384         22.0 ns         22.0 ns     31854879 bytes_per_second=10.8578Ti/s items_per_second=45.541M/s
// LongDoubleAllocate/65536         22.0 ns         22.0 ns     31874171 bytes_per_second=43.4297Ti/s items_per_second=45.5394M/s
// LongDoubleAllocate/262144        22.0 ns         22.0 ns     31864845 bytes_per_second=173.691Ti/s items_per_second=45.532M/s
// LongDoubleAllocate/1048576       22.0 ns         22.0 ns     31818965 bytes_per_second=695.019Ti/s items_per_second=45.5488M/s

// Allocate/1                7.82 ns         7.82 ns     74043189 bytes_per_second=122.025Mi/s items_per_second=127.952M/s
// Allocate/2                9.43 ns         9.43 ns     83954164 bytes_per_second=202.247Mi/s items_per_second=106.036M/s
// Allocate/4                7.79 ns         7.79 ns     90684205 bytes_per_second=489.523Mi/s items_per_second=128.325M/s
// Allocate/8                7.77 ns         7.77 ns     88781699 bytes_per_second=981.569Mi/s items_per_second=128.656M/s
// Allocate/16               7.91 ns         7.91 ns     89534014 bytes_per_second=1.88409Gi/s items_per_second=126.439M/s
// Allocate/32               7.79 ns         7.79 ns     89482380 bytes_per_second=3.82594Gi/s items_per_second=128.377M/s
// Allocate/64               7.80 ns         7.80 ns     89923628 bytes_per_second=7.64018Gi/s items_per_second=128.181M/s
// Allocate/128              7.80 ns         7.80 ns     89450912 bytes_per_second=15.284Gi/s items_per_second=128.211M/s
// Allocate/256              7.79 ns         7.79 ns     89633861 bytes_per_second=30.6175Gi/s items_per_second=128.419M/s
// Allocate/512              7.87 ns         7.87 ns     89029319 bytes_per_second=60.5854Gi/s items_per_second=127.057M/s
// Allocate/1024             7.81 ns         7.81 ns     88257513 bytes_per_second=122.128Gi/s items_per_second=128.06M/s
// Allocate/2048             19.7 ns         19.7 ns     36460273 bytes_per_second=97.0368Gi/s items_per_second=50.8752M/s
// Allocate/4096             19.3 ns         19.3 ns     36101464 bytes_per_second=197.51Gi/s items_per_second=51.776M/s
// Allocate/8192             19.3 ns         19.3 ns     36371430 bytes_per_second=394.293Gi/s items_per_second=51.6808M/s
// Allocate/16384            18.8 ns         19.3 ns     36200355 bytes_per_second=791.679Gi/s items_per_second=51.8835M/s
// Allocate/32768            22.3 ns         22.8 ns     30986715 bytes_per_second=1.30915Ti/s items_per_second=43.9279M/s
// Allocate/65536            22.5 ns         23.0 ns     30310142 bytes_per_second=2.59408Ti/s items_per_second=43.5215M/s
// Allocate/131072           22.5 ns         22.8 ns     30657638 bytes_per_second=5.21797Ti/s items_per_second=43.7715M/s
// Allocate/262144           22.5 ns         22.8 ns     31257335 bytes_per_second=10.4492Ti/s items_per_second=43.8271M/s
// Allocate/524288           22.7 ns         23.0 ns     31280850 bytes_per_second=20.7587Ti/s items_per_second=43.5341M/s
// Allocate/1048576          22.3 ns         22.5 ns     31263920 bytes_per_second=42.32Ti/s items_per_second=44.3757M/s
// Allocate/2097152          22.7 ns         22.9 ns     31070838 bytes_per_second=83.1301Ti/s items_per_second=43.5841M/s
// Allocate/4194304          22.7 ns         22.9 ns     30345883 bytes_per_second=166.521Ti/s items_per_second=43.6525M/s
// Allocate/8388608          23.1 ns         23.3 ns     30235053 bytes_per_second=328.099Ti/s items_per_second=43.0045M/s
// Allocate/16777216         22.8 ns         23.0 ns     27915970 bytes_per_second=663.979Ti/s items_per_second=43.5145M/s
// Allocate/33554432        10581 ns        10386 ns        69352 bytes_per_second=2.93823Ti/s items_per_second=96.2799k/s
// Allocate/67108864        11438 ns        11261 ns        63435 bytes_per_second=5.42004Ti/s items_per_second=88.802k/s
// Allocate/134217728       12519 ns        12330 ns        51268 bytes_per_second=9.90064Ti/s items_per_second=81.106k/s
// Allocate/268435456       13932 ns        13661 ns        49564 bytes_per_second=17.8711Ti/s items_per_second=73.2k/s
// Allocate/536870912       18501 ns        18185 ns        39615 bytes_per_second=26.8509Ti/s items_per_second=54.9906k/s
// Allocate/1073741824      25914 ns        25268 ns        26570 bytes_per_second=38.6478Ti/s items_per_second=39.5754k/s
// CharAllocate
//  11.42 │ 40:┌─→mov      %r14,%rdi                                                                                                                                         ▒
//  24.59 │    │→ call     malloc@plt                                                                                                                                        ▒
//  10.85 │    │  mov      %rax,-0x60(%rbp)                                                                                                                                  ▒
//   6.90 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  22.37 │    │→ call     free@plt                                                                                                                                          ▒
//   7.99 │    │  test     %r12,%r12                                                                                                                                         ▒
//   1.48 │    │↓ jle      1ad                                                                                                                                               ▒
//   5.77 │    │  dec      %r12                                                                                                                                              ▒
//   8.64 │    └──jne      40
// IntAllocate
//  12.38 │ 50:┌─→mov      %r14,%rdi                                                                                                                                         ▒
//  24.18 │    │→ call     malloc@plt                                                                                                                                        ◆
//   9.95 │    │  mov      %rax,-0x60(%rbp)                                                                                                                                  ▒
//   7.10 │    │  mov      %rax,%rdi                                                                                                                                         ▒
//  22.17 │    │→ call     free@plt                                                                                                                                          ▒
//   8.46 │    │  test     %r12,%r12                                                                                                                                         ▒
//   1.49 │    │↓ jle      1c1                                                                                                                                               ▒
//   5.30 │    │  dec      %r12                                                                                                                                              ▒
//   8.97 │    └──jne      50
// -   96.33%     6.53%  MallocAllocate.  MallocAllocate.exe    [.] Allocate(benchmark::State&)                                                                              ▒
//    - 89.80% Allocate(benchmark::State&)                                                                                                                                   ▒
//       + 38.37% malloc                                                                                                                                                     ▒
//       + 34.45% cfree@GLIBC_2.2.5                                                                                                                                          ▒
//       + 11.22% __munmap

// Note1: both g++ and clang version seems to get same result and both better than using new operatior
// Note2: As the memory allocate more than or equal to 32MB, the program use another method. __munmap
BENCHMARK_MAIN();