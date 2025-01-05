#include <benchmark/benchmark.h>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile char c = 'a';
  }
}

static void ShortAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile short s = 69;
  }
}

static void IntAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile int i = 69;
  }
}

static void LongLongAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile long long l = 69;
  }
}

static void FloatAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile float f = 1.6;
  }
}

static void DoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile double d = 1.660;
  }
}

static void LongDoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    volatile long double ld = 9.4444332;
  }
}

BENCHMARK(CharAssignment); 
BENCHMARK(ShortAssignment); 
BENCHMARK(IntAssignment); 
BENCHMARK(LongLongAssignment); 
BENCHMARK(FloatAssignment); 
BENCHMARK(DoubleAssignment); 
BENCHMARK(LongDoubleAssignment); 
// g++
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAssignment            0.242 ns        0.242 ns   2898570336
// ShortAssignment           0.243 ns        0.243 ns   2878482671
// IntAssignment             0.242 ns        0.242 ns   2890150559
// LongLongAssignment        0.242 ns        0.242 ns   2888058391
// FloatAssignment           0.242 ns        0.242 ns   2888382744
// DoubleAssignment          0.242 ns        0.242 ns   2888990845
// LongDoubleAssignment       3.16 ns         3.16 ns    221794885
// CharAssigment
//  14.82 │60:┌─→movb    $0x61,-0x11(%rbp)                                                                                                                                      ▒
//  13.89 │   │  movb    $0x61,-0x11(%rbp)                                                                                                                                      ▒
//  11.11 │   │  sub     $0x2,%rax                                                                                                                                              ▒
//  11.08 │   │↑ je      1e                                                                                                                                                     ▒
//  13.35 │   │  movb    $0x61,-0x11(%rbp)                                                                                                                                      ▒
//  14.21 │   │  movb    $0x61,-0x11(%rbp)                                                                                                                                      ▒
//  10.93 │   │  sub     $0x2,%rax                                                                                                                                              ◆
//  10.62 │   └──jne     60
//  ShrotAssignment
// 78:   mov     $0x45,%edx
//  18.35 │      mov     $0x45,%ecx
//  18.58 │      mov     %dx,-0x12(%rbp)
//  14.56 │      mov     %cx,-0x12(%rbp)
//  18.01 │      sub     $0x2,%rax
//  16.34 │    ↑ jne     78
// LongDoubleAssignment
//  16.46 │80:   fstpt   -0x20(%rbp)                                                                                                                                            ▒
//   8.27 │      fldt    -0x20(%rbp)                                                                                                                                            ▒
//  17.34 │      fstpt   -0x20(%rbp)                                                                                                                                            ▒
//   7.49 │      fldt    -0x20(%rbp)                                                                                                                                            ▒
//   0.81 │      sub     $0x2,%rax                                                                                                                                              ▒
//   0.81 │    ↑ je      28                                                                                                                                                     ▒
//  14.74 │      fstpt   -0x20(%rbp)                                                                                                                                            ▒
//   6.99 │      fldt    -0x20(%rbp)                                                                                                                                            ▒
//  17.62 │      fstpt   -0x20(%rbp)                                                                                                                                            ▒
//   7.66 │      fldt    -0x20(%rbp)                                                                                                                                            ◆
//   0.81 │      sub     $0x2,%rax                                                                                                                                              ▒
//   1.00 │    ↑ jne     80


// clang++
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAssignment            0.238 ns        0.238 ns   2946215241
// ShortAssignment           0.238 ns        0.238 ns   2944723167
// IntAssignment             0.238 ns        0.238 ns   2940730431
// LongLongAssignment        0.238 ns        0.238 ns   2940978226
// FloatAssignment           0.238 ns        0.238 ns   2940407690
// DoubleAssignment          0.238 ns        0.238 ns   2942881660
// LongDoubleAssignment       1.20 ns         1.20 ns    580229132
// CharAssignment
// 16.66 │30:   movb   $0x61,-0x19(%rbp)                                                                                                                                       ▒
// 33.34 │      test   %r14,%r14                                                                                                                                               ▒
//       │    ↓ jle    50                                                                                                                                                      ▒
// 33.34 │      dec    %r14                                                                                                                                                    ▒
// 16.66 │    ↑ jne    30
// ShortAssignment
// 16.64 │30:   movw   $0x45,-0x1a(%rbp)
// 33.36 │      test   %r14,%r14
//       │    ↓ jle    52
// 33.28 │      dec    %r14
// 16.72 │    ↑ jne    30
// LongDoubleAssignment
//  2.94 │40:   fld    %st(0)
// 81.40 │      fstpt  -0x30(%rbp)
//  3.18 │      test   %r14,%r14
//       │    ↓ jle    63
//  6.09 │      dec    %r14
//  6.39 │    ↑ jne    40


// Note1: The effect of unstable time with superscaler problem from thet local allocate disappear as the loop iis unroll and forced to be 2 cycles per 2 loop.
// Note2: The clang++ time become better as the mov instruction from access is disappear making it exact 4 instructions.
BENCHMARK_MAIN();