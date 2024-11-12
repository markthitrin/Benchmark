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
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    short* p = (short*)calloc(array_size,sizeof(int));
    escape(p);
    free(p);
  }
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    int* p = (int*)calloc(array_size,sizeof(int));
    escape(p);
    free(p);
  }
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long long* p = (long long*)calloc(array_size,sizeof(long long));
    escape(p);
    free(p);
  }
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    float* p = (float*)calloc(array_size,sizeof(float));
    escape(p);
    free(p);
  }
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    double* p = (double*)calloc(array_size,sizeof(double));
    escape(p);
    free(p);
  }
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long double* p = (long double*)calloc(array_size,sizeof(long double));
    escape(p);
    free(p);
  }
}

static void Allocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = (char*)calloc(array_size,sizeof(char));
    escape(p);
    escape(p + (1 << 20));
    free(p);
  }
}

BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<10); 
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// CharAllocate/1                13.2 ns         13.2 ns     53136076
// CharAllocate/4                13.2 ns         13.2 ns     53169962
// CharAllocate/16               13.2 ns         13.2 ns     53126169
// CharAllocate/64               13.6 ns         13.6 ns     50859850
// CharAllocate/256              40.4 ns         40.4 ns     17325373
// CharAllocate/1024             40.7 ns         40.7 ns     17148454

BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// ShortAllocate/1               13.2 ns         13.2 ns     53151379
// ShortAllocate/4               13.2 ns         13.2 ns     53086469
// ShortAllocate/16              13.7 ns         13.7 ns     51016356
// ShortAllocate/64              39.7 ns         39.7 ns     17629480
// ShortAllocate/256             42.3 ns         42.3 ns     16427636
// ShortAllocate/1024            60.7 ns         60.7 ns     11109984

BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// IntAllocate/1                 13.2 ns         13.2 ns     53024180
// IntAllocate/4                 13.2 ns         13.2 ns     53144094
// IntAllocate/16                13.7 ns         13.7 ns     51159134
// IntAllocate/64                40.6 ns         40.6 ns     17349312
// IntAllocate/256               40.7 ns         40.7 ns     17201950
// IntAllocate/1024              60.2 ns         60.2 ns     11610199

BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// LongLongAllocate/1            13.2 ns         13.2 ns     53109281
// LongLongAllocate/4            13.4 ns         13.4 ns     52174630
// LongLongAllocate/16           40.9 ns         40.9 ns     17419789
// LongLongAllocate/64           42.2 ns         42.1 ns     17058107
// LongLongAllocate/256          48.1 ns         48.1 ns     15045599
// LongLongAllocate/1024         91.4 ns         91.4 ns      7710654

BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// FloatAllocate/1               13.3 ns         13.4 ns     52684837
// FloatAllocate/4               13.3 ns         13.3 ns     52664751
// FloatAllocate/16              13.9 ns         13.9 ns     50811241
// FloatAllocate/64              40.0 ns         40.0 ns     17647953
// FloatAllocate/256             40.6 ns         40.6 ns     16929605
// FloatAllocate/1024            61.2 ns         61.2 ns      8515034

BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// DoubleAllocate/1              13.3 ns         13.3 ns     52372375
// DoubleAllocate/4              13.6 ns         13.6 ns     51416033
// DoubleAllocate/16             40.4 ns         40.4 ns     17421690
// DoubleAllocate/64             41.4 ns         41.4 ns     16658200
// DoubleAllocate/256            47.4 ns         47.4 ns     14905500
// DoubleAllocate/1024           90.9 ns         90.9 ns      7656760


BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// LongDoubleAllocate/1          13.4 ns         13.4 ns     51476951
// LongDoubleAllocate/4          13.9 ns         13.9 ns     50188107
// LongDoubleAllocate/16         41.1 ns         41.1 ns     16610610
// LongDoubleAllocate/64         41.4 ns         41.4 ns     14046461
// LongDoubleAllocate/256        60.5 ns         60.5 ns     11509770
// LongDoubleAllocate/1024        150 ns          150 ns      4684631

BENCHMARK(Allocate)->RangeMultiplier(2)->Range(1,1<<30);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// Allocate/1                    13.3 ns         13.3 ns     52683562
// Allocate/2                    13.3 ns         13.3 ns     52643993
// Allocate/4                    13.5 ns         13.5 ns     52715449
// Allocate/8                    13.4 ns         13.4 ns     52780576
// Allocate/16                   13.3 ns         13.3 ns     52776756
// Allocate/32                   13.3 ns         13.3 ns     52573685
// Allocate/64                   13.8 ns         13.8 ns     44059784
// Allocate/128                  40.4 ns         40.4 ns     17379700
// Allocate/256                  40.2 ns         40.2 ns     17475085
// Allocate/512                  40.9 ns         40.9 ns     17246388
// Allocate/1024                 41.6 ns         41.6 ns     16844176
// Allocate/2048                 46.8 ns         46.8 ns     15157360
// Allocate/4096                 60.7 ns         60.7 ns     11364315
// Allocate/8192                 89.9 ns         89.9 ns      7749915
// Allocate/16384                 150 ns          150 ns      4643583
// Allocate/32768                 284 ns          284 ns      2483530
// Allocate/65536                 549 ns          549 ns      1265964
// Allocate/131072               1064 ns         1064 ns       655452
// Allocate/262144               2135 ns         2135 ns       334115
// Allocate/524288               5321 ns         5322 ns       128785
// Allocate/1048576             11964 ns        11965 ns        55487
// Allocate/2097152             24638 ns        24638 ns        28382
// Allocate/4194304            249357 ns       249346 ns         2743
// Allocate/8388608           1551092 ns      1550733 ns          429
// Allocate/16777216          2967111 ns      2966969 ns          235
// Allocate/33554432            11809 ns        11805 ns        59287
// Allocate/67108864            12238 ns        12236 ns        56483
// Allocate/134217728           13104 ns        13104 ns        53117
// Allocate/268435456           14797 ns        14798 ns        46987
// Allocate/536870912           19150 ns        19145 ns        35965
// Allocate/1073741824          26464 ns        26429 ns        26833

// For large memory requests (often over 128 KB, though this threshold can vary), 
// calloc in glibc typically does not use malloc in the traditional sense. Instead, 
// it switches to memory-mapped pages by using the mmap system call with flags like MAP_ANONYMOUS 
// and MAP_PRIVATE.

// The mmap system call requests memory directly from the kernel, bypassing the memory pool 
// usually managed by malloc. This allows calloc to access large contiguous memory blocks directly from the OS.

// Zero-Initialization with mmap: When mmap is used with MAP_ANONYMOUS, 
// the kernel automatically provides a zero-initialized memory region, effectively skipping the need 
// for a memset call. This can make calloc for large allocations faster, as it avoids redundant zeroing operations.

BENCHMARK_MAIN();