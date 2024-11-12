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
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    short* p = (short*)malloc(sizeof(int)*array_size);
    escape(p);
    free(p);
  }
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    int* p = (int*)malloc(sizeof(int)*array_size);
    escape(p);
    free(p);
  }
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long long* p = (long long*)malloc(sizeof(long long)*array_size);
    escape(p);
    free(p);
  }
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    float* p = (float*)malloc(sizeof(float)*array_size);
    escape(p);
    free(p);
  }
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    double* p = (double*)malloc(sizeof(double)*array_size);
    escape(p);
    free(p);
  }
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    long double* p = (long double*)malloc(sizeof(long double)*array_size);
    escape(p);
    free(p);
  }
}

static void Allocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    char* p = (char*)malloc(sizeof(char)*array_size);;
    escape(p);
    free(p);
  }
}

BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<10); 
// ------------------------------------------------------------
// Benchmark                  Time             CPU   Iterations
// ------------------------------------------------------------
// CharAllocate/1          8.03 ns         8.03 ns     89191334
// CharAllocate/4          8.04 ns         8.04 ns     83692100
// CharAllocate/16         7.92 ns         7.91 ns     87147432
// CharAllocate/64         7.86 ns         7.86 ns     87982498
// CharAllocate/256        7.81 ns         7.80 ns     89324051
// CharAllocate/1024       8.19 ns         8.18 ns     87676306

BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------
// Benchmark                  Time             CPU   Iterations 
// ------------------------------------------------------------
// CharAllocate/1          8.03 ns         8.03 ns     89191334
// CharAllocate/4          8.04 ns         8.04 ns     83692100
// CharAllocate/16         7.92 ns         7.91 ns     87147432
// CharAllocate/64         7.86 ns         7.86 ns     87982498
// CharAllocate/256        7.81 ns         7.80 ns     89324051
// CharAllocate/1024       8.19 ns         8.18 ns     87676306

BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// -----------------------------------------------------------
// Benchmark                 Time             CPU   Iterations
// -----------------------------------------------------------
// IntAllocate/1          7.66 ns         7.66 ns     90975553
// IntAllocate/4          7.79 ns         7.79 ns     88771634
// IntAllocate/16         7.90 ns         7.90 ns     88804151
// IntAllocate/64         7.83 ns         7.82 ns     88951852
// IntAllocate/256        7.81 ns         7.81 ns     89331098
// IntAllocate/1024       19.3 ns         19.3 ns     36521101

BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// LongLongAllocate/1          7.97 ns         7.97 ns     89752014
// LongLongAllocate/4          7.83 ns         7.83 ns     85727129
// LongLongAllocate/16         7.83 ns         7.83 ns     88644680
// LongLongAllocate/64         7.95 ns         7.95 ns     88909631
// LongLongAllocate/256        29.3 ns         29.3 ns     23773240
// LongLongAllocate/1024       19.2 ns         19.2 ns     36361501

BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// -------------------------------------------------------------
// Benchmark                   Time             CPU   Iterations
// -------------------------------------------------------------
// FloatAllocate/1          7.98 ns         7.98 ns     87440689
// FloatAllocate/4          7.86 ns         7.86 ns     86636805
// FloatAllocate/16         7.80 ns         7.80 ns     89212027
// FloatAllocate/64         8.37 ns         8.37 ns     89580323
// FloatAllocate/256        8.30 ns         8.30 ns     79390894
// FloatAllocate/1024       19.9 ns         19.9 ns     34506335

BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// --------------------------------------------------------------
// Benchmark                    Time             CPU   Iterations
// --------------------------------------------------------------
// DoubleAllocate/1          8.08 ns         8.09 ns     71056857
// DoubleAllocate/4          7.90 ns         7.91 ns     87654626
// DoubleAllocate/16         7.84 ns         7.85 ns     86536806
// DoubleAllocate/64         7.87 ns         7.88 ns     86302202
// DoubleAllocate/256        29.4 ns         29.5 ns     24005721
// DoubleAllocate/1024       19.4 ns         19.4 ns     35932889


BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<10);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// LongDoubleAllocate/1          7.85 ns         7.85 ns     89074317
// LongDoubleAllocate/4          7.81 ns         7.81 ns     88954876
// LongDoubleAllocate/16         7.80 ns         7.80 ns     88086591
// LongDoubleAllocate/64         7.84 ns         7.84 ns     88479369
// LongDoubleAllocate/256        19.2 ns         19.2 ns     36440274
// LongDoubleAllocate/1024       19.2 ns         19.2 ns     35845108

BENCHMARK(Allocate)->RangeMultiplier(2)->Range(1,1<<30);
// --------------------------------------------------------------
// Benchmark                    Time             CPU   Iterations
// --------------------------------------------------------------
// Allocate/1                7.85 ns         7.85 ns     86090723
// Allocate/2                7.73 ns         7.73 ns     90144071
// Allocate/4                7.71 ns         7.71 ns     89847220
// Allocate/8                7.70 ns         7.71 ns     90417548
// Allocate/16               7.84 ns         7.83 ns     88448500
// Allocate/32               7.86 ns         7.86 ns     84658087
// Allocate/64               7.82 ns         7.82 ns     87283858
// Allocate/128              8.23 ns         8.22 ns     89015237
// Allocate/256              8.04 ns         8.04 ns     84598811
// Allocate/512              7.97 ns         7.98 ns     87659882
// Allocate/1024             8.06 ns         8.06 ns     88422609
// Allocate/2048             19.8 ns         19.8 ns     36047072
// Allocate/4096             19.5 ns         19.5 ns     35915185
// Allocate/8192             19.8 ns         19.8 ns     35822005
// Allocate/16384            19.8 ns         19.8 ns     35105499
// Allocate/32768            22.8 ns         22.8 ns     30654225
// Allocate/65536            22.9 ns         22.9 ns     30660555
// Allocate/131072           23.4 ns         23.4 ns     30668781
// Allocate/262144           22.4 ns         22.4 ns     30574026
// Allocate/524288           22.6 ns         22.6 ns     32078492
// Allocate/1048576          22.2 ns         22.2 ns     31882847
// Allocate/2097152          22.3 ns         22.4 ns     31891951
// Allocate/4194304          22.4 ns         22.4 ns     30295003
// Allocate/8388608          22.4 ns         22.4 ns     31776827
// Allocate/16777216         22.7 ns         22.7 ns     31047436
// Allocate/33554432        13995 ns        13999 ns        53433
// Allocate/67108864        13268 ns        13239 ns        52130
// Allocate/134217728       14077 ns        14083 ns        49545
// Allocate/268435456       16461 ns        16459 ns        44402
// Allocate/536870912       20138 ns        19910 ns        34743
// Allocate/1073741824      27689 ns        26966 ns        25941


BENCHMARK_MAIN();