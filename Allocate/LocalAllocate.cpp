#include <benchmark/benchmark.h>
#define MCA_START __asm volatile("# LLVM-MCA-BEGIN");
#define MCA_END __asm volatile("# LLVM-MCA-END");

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAssignment(benchmark::State& state) {
  for(auto _ : state) {
    char c = 'a';
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
}

static void ShortAssignment(benchmark::State& state) {
  for(auto _ : state) {
    short s = 69;
    escape(&s);
  }
  state.SetItemsProcessed(state.iterations());
}

static void IntAssignment(benchmark::State& state) {
  for(auto _ : state) {
    MCA_START
    int i = 69;
    escape(&i);
    MCA_END
  }
  state.SetItemsProcessed(state.iterations());
}

static void LongLongAssignment(benchmark::State& state) {
  for(auto _ : state) {
    long long l = 69;
    escape(&l);
  }
  state.SetItemsProcessed(state.iterations());
}

static void FloatAssignment(benchmark::State& state) {
  for(auto _ : state) {
    float f = 1.6;
    escape(&f);
  }
  state.SetItemsProcessed(state.iterations());
}

static void DoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    double d = 1.660;
    escape(&d);
  }
  state.SetItemsProcessed(state.iterations());
}

static void LongDoubleAssignment(benchmark::State& state) {
  for(auto _ : state) {
    long double ld = 9.4444332;
    escape(&ld);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(CharAssignment);
BENCHMARK(ShortAssignment);
BENCHMARK(IntAssignment);
BENCHMARK(LongLongAssignment);
BENCHMARK(FloatAssignment);
BENCHMARK(DoubleAssignment); 
BENCHMARK(LongDoubleAssignment);
// g++
// -------------------------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------
// CharAssignment            0.239 ns        0.239 ns   2923104585 items_per_second=4.17614G/s
// ShortAssignment           0.240 ns        0.240 ns   2923174530 items_per_second=4.17448G/s
// IntAssignment             0.239 ns        0.239 ns   2923704743 items_per_second=4.17729G/s
// LongLongAssignment        0.239 ns        0.239 ns   2924893588 items_per_second=4.17787G/s
// FloatAssignment           0.479 ns        0.479 ns   1461560813 items_per_second=2.08691G/s
// DoubleAssignment          0.240 ns        0.239 ns   2923500947 items_per_second=4.17558G/s
// LongDoubleAssignment       3.13 ns         3.13 ns    223571291 items_per_second=319.409M/s
// CharAssignemnnt
// 33.34 │110:┌─→movb     $0x61,-0x51(%rbp)                                                                                                                                    ▒
// 33.32 │    │  sub      $0x1,%rax                                                                                                                                            ▒
// 33.34 │    └──jne      110     
// IntAssigment
// 33.31 │110:   movl     $0x45,-0x54(%rbp)
// 33.36 │       sub      $0x1,%rax
// 33.33 │     ↑ jne      110
// FloatAssigment
// 49.32 │118:┌─→movss    %xmm0,-0x54(%rbp)                                                                                                                                    ▒
// 26.44 │    │  sub      $0x1,%rax                                                                                                                                            ▒
// 24.24 │    └──jne      118       
// LongDoubleAssignment
// 62.74 │120:   fstpt    -0x60(%rbp)                                                                                                                                          ▒
// 30.21 │       fldt     -0x60(%rbp)                                                                                                                                          ▒
//  3.52 │       sub      $0x1,%rax                                                                                                                                            ▒
//  3.52 │     ↑ jne      120 

                                                                                                                                            
// clang++
// -------------------------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------
// CharAssignment             1.20 ns         1.20 ns    583997387 items_per_second=835.109M/s
// ShortAssignment            1.20 ns         1.20 ns    584068515 items_per_second=835.368M/s
// IntAssignment             0.484 ns        0.484 ns   1439973586 items_per_second=2.06566G/s
// LongLongAssignment        0.484 ns        0.484 ns   1446735889 items_per_second=2.06725G/s
// FloatAssignment           0.483 ns        0.483 ns   1446781599 items_per_second=2.06853G/s
// DoubleAssignment          0.483 ns        0.483 ns   1450222727 items_per_second=2.07177G/s
// LongDoubleAssignment       1.45 ns         1.45 ns    480733800 items_per_second=687.959M/s
// CharAssignment
//  33.32 │ 30:   movb     $0x61,-0x38(%rbp)                                                                                                                                    ▒
//  16.63 │       mov      %rax,-0x50(%rbp)                                                                                                                                     ▒
//  16.71 │       test     %r14,%r14                                                                                                                                            ▒
//        │     ↓ jle      f7                                                                                                                                                   ▒
//  16.69 │       dec      %r14                                                                                                                                                 ▒
//  16.66 │     ↑ jne      30
// IntAssignment
//  16.68 │ 30:   movl     $0x45,-0x38(%rbp)
//  16.68 │       mov      %rax,-0x50(%rbp)
//  16.71 │       test     %r14,%r14
//        │     ↓ jle      fa
//  33.29 │       dec      %r14
//  16.63 │     ↑ jne      30
// LongDoubleAssignment
//   2.79 │ 40:┌─→fld      %st(0)                                                                                                                                               ▒
//  81.61 │    │  fstpt    -0x40(%rbp)                                                                                                                                          ▒
//   3.08 │    │  mov      %rax,-0x58(%rbp)                                                                                                                                     ▒
//   3.20 │    │  test     %r14,%r14                                                                                                                                            ▒
//        │    │↓ jle      10a                                                                                                                                                  ▒
//   3.14 │    │  dec      %r14                                                                                                                                                 ▒
//   6.19 │    └──jne      40      

// Note1: The affect of the superscalar is seems to be occur, In g++ version, some loop might take longer than expected.
// Note2: clang++ seems to use alternative approach which use 2 cycles per loop. The test is for jump to the end loop if it is zero.
//        and the second jmp jumping to the start. It seems that this approach is better when the instructions is long enoug like
//        LongDoubleAssigment case where it is the only one that is faster than g++. The approach is just do while loop like since it
//        yeild less jmp.

BENCHMARK_MAIN();