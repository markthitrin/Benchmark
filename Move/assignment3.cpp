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
  delete a;
  delete c;
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
  delete a;
  delete c;
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
  delete a;
  delete c;
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
  delete a;
  delete c;
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
  delete a;
  delete c;
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
  delete a;
  delete c;
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
  delete a;
  delete c;
}

// BENCHMARK(CharAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
// -----------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations
// -----------------------------------------------------------------
// CharAssignment/1             2.50 ns         2.50 ns    269099671
// CharAssignment/4             2.87 ns         2.87 ns    245753108
// CharAssignment/16            2.26 ns         2.26 ns    308948768
// CharAssignment/64            2.28 ns         2.28 ns    307735817
// CharAssignment/256           4.02 ns         4.02 ns    305552435
// CharAssignment/1024          8.12 ns         8.12 ns     75216478
// CharAssignment/4096          32.1 ns         32.1 ns     21713957
// CharAssignment/16384          140 ns          140 ns      4975007
// CharAssignment/65536         1016 ns         1016 ns       688830
// CharAssignment/262144        4705 ns         4705 ns       153393
// CharAssignment/1048576      22874 ns        22871 ns        33922

// BENCHMARK(ShortAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// ShortAssignment/1             2.19 ns         2.19 ns    319486508
// ShortAssignment/4             2.19 ns         2.19 ns    315058019
// ShortAssignment/16            2.20 ns         2.20 ns    318001702
// ShortAssignment/64            2.20 ns         2.20 ns    317688168
// ShortAssignment/256           4.71 ns         4.71 ns    125963202
// ShortAssignment/1024          17.0 ns         17.0 ns     41025518
// ShortAssignment/4096          63.0 ns         63.0 ns     11017978
// ShortAssignment/16384          514 ns          514 ns      1355047
// ShortAssignment/65536         2033 ns         2033 ns       342437
// ShortAssignment/262144        9741 ns         9740 ns        69822
// ShortAssignment/1048576     164471 ns       164447 ns         4274

// BENCHMARK(IntAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// IntAssignment/1             2.55 ns         2.55 ns    258593928
// IntAssignment/4             2.38 ns         2.38 ns    293321497
// IntAssignment/16            1.77 ns         1.77 ns    396353197
// IntAssignment/64            4.07 ns         4.07 ns    308612985
// IntAssignment/256           8.23 ns         8.23 ns     74436355
// IntAssignment/1024          32.6 ns         32.6 ns     21457625
// IntAssignment/4096           142 ns          142 ns      4930395
// IntAssignment/16384         1026 ns         1026 ns       683521
// IntAssignment/65536         4558 ns         4557 ns       154129
// IntAssignment/262144       20318 ns        20315 ns        33487
// IntAssignment/1048576     374571 ns       374501 ns         1923

// BENCHMARK(LongLongAssignment)->RangeMultiplier(4)->Range(1, 1<<20); 
// ---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
// ---------------------------------------------------------------------
// LongLongAssignment/1             2.23 ns         2.23 ns    313566593
// LongLongAssignment/4             2.23 ns         2.23 ns    313536284
// LongLongAssignment/16            2.24 ns         2.24 ns    312859929
// LongLongAssignment/64            5.71 ns         5.71 ns    126778556
// LongLongAssignment/256           17.0 ns         17.0 ns     41092199
// LongLongAssignment/1024          63.1 ns         63.1 ns     11049622
// LongLongAssignment/4096           516 ns          516 ns      1346916
// LongLongAssignment/16384         2039 ns         2039 ns       342464
// LongLongAssignment/65536         9842 ns         9841 ns        67900
// LongLongAssignment/262144      162415 ns       162385 ns         4448
// LongLongAssignment/1048576    1169574 ns      1169412 ns          591

// BENCHMARK(FloatAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
// ------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations
// ------------------------------------------------------------------
// FloatAssignment/1             2.22 ns         2.22 ns    314856653
// FloatAssignment/4             2.22 ns         2.22 ns    314615006
// FloatAssignment/16            2.23 ns         2.23 ns    314321857
// FloatAssignment/64            4.05 ns         4.05 ns    310563426
// FloatAssignment/256           9.36 ns         9.36 ns     85157213
// FloatAssignment/1024          32.4 ns         32.4 ns     21592645
// FloatAssignment/4096           143 ns          143 ns      4891198
// FloatAssignment/16384         1031 ns         1031 ns       680058
// FloatAssignment/65536         4579 ns         4578 ns       155913
// FloatAssignment/262144       19827 ns        19825 ns        33089
// FloatAssignment/1048576     299582 ns       299520 ns         2248

// BENCHMARK(DoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
// -------------------------------------------------------------------
// Benchmark                         Time             CPU   Iterations
// -------------------------------------------------------------------
// DoubleAssignment/1             2.24 ns         2.24 ns    312494852
// DoubleAssignment/4             1.77 ns         1.77 ns    372110853
// DoubleAssignment/16            2.26 ns         2.26 ns    310035221
// DoubleAssignment/64            5.56 ns         5.56 ns    125931511
// DoubleAssignment/256           17.2 ns         17.1 ns     40814891
// DoubleAssignment/1024          63.6 ns         63.6 ns     10998615
// DoubleAssignment/4096           516 ns          516 ns      1348938
// DoubleAssignment/16384         2047 ns         2046 ns       340164
// DoubleAssignment/65536         9717 ns         9716 ns        67239
// DoubleAssignment/262144      160943 ns       160905 ns         4360
// DoubleAssignment/1048576    1182257 ns      1181996 ns          580

// BENCHMARK(LongDoubleAssignment)->RangeMultiplier(4)->Range(1, 1<<20);
// -----------------------------------------------------------------------
// Benchmark                             Time             CPU   Iterations
// -----------------------------------------------------------------------
// LongDoubleAssignment/1             2.25 ns         2.25 ns    310813270
// LongDoubleAssignment/4             1.79 ns         1.79 ns    392047379
// LongDoubleAssignment/16            4.03 ns         4.03 ns    307891021
// LongDoubleAssignment/64            13.8 ns         13.8 ns     86004621
// LongDoubleAssignment/256           31.4 ns         31.4 ns     22253847
// LongDoubleAssignment/1024           141 ns          141 ns      4944353
// LongDoubleAssignment/4096          1020 ns         1019 ns       685356
// LongDoubleAssignment/16384         4564 ns         4564 ns       157799
// LongDoubleAssignment/65536        20511 ns        20509 ns        36157
// LongDoubleAssignment/262144      353205 ns       353134 ns         1983
// LongDoubleAssignment/1048576    2397853 ns      2397528 ns          280

// memcpy with 512 bytes


BENCHMARK_MAIN();