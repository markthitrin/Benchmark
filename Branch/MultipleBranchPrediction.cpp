#include <benchmark/benchmark.h>
#include <stdlib.h>

constexpr int array_size = (1 << 20);

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void AlwaysTrueLogicalBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  bool* d = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
    d[i] = !c[i];
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      if(c[i] || d[i]) {
        a[i] = a[i] + b[i];
      }
      else {
        a[i] = a[i] * b[i];
      }
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

static void RandomLogicalBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  bool* d = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
    d[i] = std::rand() & 1;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      if(c[i] || d[i]) {
        a[i] = a[i] + b[i];
      }
      else {
        a[i] = a[i] * b[i];
      }
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

static void AlwaysTrueBitwiseBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  bool* d = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
    d[i] = !c[i];
  }
  for(auto _ : state) {
    char* p1 = (char*) c;
    char* p2 = (char*) d;
    for(int i = 0;i < array_size;i++) {
      if(*(p1++) | *(p2++)) {
        a[i] = a[i] + b[i];
      }
      else {
        a[i] = a[i] * b[i];
      }
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

static void RandomBitwiseBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  bool* d = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
    d[i] = std::rand() & 1;
  }
  for(auto _ : state) {
    char* p1 = (char*) c;
    char* p2 = (char*) d;
    for(int i = 0;i < array_size;i++) {
      if(*(p1++) | *(p2++)) {
        a[i] = a[i] + b[i];
      }
      else {
        a[i] = a[i] * b[i];
      }
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(AlwaysTrueLogicalBranch);
BENCHMARK(RandomLogicalBranch);
BENCHMARK(AlwaysTrueBitwiseBranch);
BENCHMARK(RandomBitwiseBranch);


// g++ (if(c[i] | d[i]) got recognized as logical or)
// ----------------------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------
// AlwaysTrueLogicalBranch    3935253 ns      3937843 ns          178 items_per_second=253.946/s
// RandomLogicalBranch        5367678 ns      5373056 ns          130 items_per_second=186.114/s
// AlwaysTrueBitwiseBranch    1183328 ns      1184494 ns          590 items_per_second=844.242/s
// RandomBitwiseBranch        1004942 ns      1005889 ns          689 items_per_second=994.146/s

// AlwaysTrueLogicalBranch
//  34.48 │280:   cmpb     $0x0,(%rbx,%rax,1)                                                                                                                                ▒
//   0.50 │     ↓ jne      2a9                                                                                                                                               ▒
//        │       imul     %ecx,%edx                                                                                                                                         ▒
//        │       mov      %edx,(%r12,%rax,4)                                                                                                                                ▒
//        │       add      $0x1,%rax                                                                                                                                         ▒
//        │       cmp      $0x100000,%rax                                                                                                                                    ▒
//        │     ↓ je       2bb                                                                                                                                               ▒
//   4.38 │299:┌─→cmpb     $0x0,0x0(%r13,%rax,1)                                                                                                                             ▒
//   6.85 │    │  mov      (%r12,%rax,4),%edx                                                                                                                                ▒
//   7.26 │    │  mov      (%r15,%rax,4),%ecx                                                                                                                                ▒
//   4.94 │    │↑ je       280                                                                                                                                               ▒
//   6.11 │2a9:│  add      %ecx,%edx                                                                                                                                         ▒
//  12.74 │    │  mov      %edx,(%r12,%rax,4)                                                                                                                                ▒
//  12.54 │    │  add      $0x1,%rax                                                                                                                                         ▒
//   6.28 │    ├──cmp      $0x100000,%rax                                                                                                                                    ▒
//        │    └──jne      299       
// AlwaysTrueBitWiseBranch
//   0.24 │288:   movdqu    (%r14,%rdx,1),%xmm13                                                                                                                             ▒
//   0.35 │       movdqa    %xmm5,%xmm10                                                                                                                                     ▒
//   0.12 │       movdqu    (%rax),%xmm12                                                                                                                                    ▒
//   0.53 │       add       $0x40,%rax                                                                                                                                       ▒
//   0.27 │       movdqu    0x0(%r13,%rdx,1),%xmm0                                                                                                                           ◆
//   0.33 │       movdqu    (%r15,%rdx,4),%xmm4                                                                                                                              ▒
//   0.24 │       movdqu    0x10(%r15,%rdx,4),%xmm3                                                                                                                          ▒
//   0.47 │       movdqu    -0x30(%rax),%xmm11                                                                                                                               ▒
//   0.32 │       por       %xmm13,%xmm0                                                                                                                                     ▒
//   0.38 │       movdqa    %xmm4,%xmm15                                                                                                                                     ▒
//   0.24 │       movdqa    %xmm6,%xmm13                                                                                                                                     ▒
//   0.27 │       movdqu    0x20(%r15,%rdx,4),%xmm2                                                                                                                          ▒
//   0.59 │       pcmpeqb   %xmm5,%xmm0  
// ...
//    0.56 │    │  psrlq     $0x20,%xmm1                                                                                                                                      ▒
//   0.94 │    │  por       %xmm12,%xmm2                                                                                                                                     ▒
//   2.81 │    │  psrlq     $0x20,%xmm7                                                                                                                                      ▒
//   5.97 │    │  movups    %xmm2,-0x20(%rax)                                                                                                                                ▒
//  10.11 │    │  pmuludq   %xmm7,%xmm1                                                                                                                                      ▒
//  13.18 │    │  pshufd    $0x8,%xmm3,%xmm3                                                                                                                                 ▒
//  15.20 │    │  pshufd    $0x8,%xmm1,%xmm1                                                                                                                                 ▒
//  11.15 │    │  punpckldq %xmm1,%xmm3                                                                                                                                      ▒
//   6.98 │    │  movdqa    %xmm8,%xmm1                                                                                                                                      ▒
//   3.19 │    │  pand      %xmm0,%xmm1                                                                                                                                      ▒
//   1.03 │    │  pandn     %xmm3,%xmm0                                                                                                                                      ▒
//   0.18 │    │  por       %xmm1,%xmm0                                                                                                                                      ▒
//   0.15 │    │  movups    %xmm0,-0x10(%rax)                                                                                                                                ▒
//   0.29 │    ├──cmp       $0x100000,%rdx                                                                                                                                   ▒
//        │    └──jne       288  

// clang++
// ----------------------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------
// AlwaysTrueLogicalBranch    3630050 ns      3628213 ns          193 items_per_second=275.618/s
// RandomLogicalBranch        5373682 ns      5370749 ns          129 items_per_second=186.194/s
// AlwaysTrueBitwiseBranch    1154846 ns      1154154 ns          588 items_per_second=866.435/s
// RandomBitwiseBranch        1162980 ns      1162394 ns          641 items_per_second=860.294/s

// RandomLogicalBranch
//        │ b0:┌─→xor      %eax,%eax                                                                                                                                         ▒
//        │    │↓ jmp      d3                                                                                                                                                ▒
//        │    │  data16   data16 cs nopw 0x0(%rax,%rax,1)                                                                                                                   ▒
//  13.12 │ c0:│  mov      (%r15,%rax,4),%ecx                                                                                                                                ▒
//  14.27 │    │  add      %ecx,(%r14,%rax,4)                                                                                                                                ▒
//  17.84 │ c8:│  inc      %rax                                                                                                                                              ▒
//   8.39 │    │  cmp      $0x100000,%rax                                                                                                                                    ▒
//        │    │↓ je       100                                                                                                                                               ▒
//   7.11 │ d3:│  cmpb     $0x0,(%r12,%rax,1)                                                                                                                                ▒
//   0.46 │    │↑ jne      c0                                                                                                                                                ◆
//  11.40 │    │  cmpb     $0x1,0x0(%r13,%rax,1)                                                                                                                             ▒
//  10.00 │    │↑ je       c0                                                                                                                                                ▒
//   2.45 │    │  mov      (%r15,%rax,4),%ecx                                                                                                                                ▒
//   4.61 │    │  imul     (%r14,%rax,4),%ecx                                                                                                                                ▒
//   3.86 │    │  mov      %ecx,(%r14,%rax,4)                                                                                                                                ▒
//   2.02 │    │↑ jmp      c8                                                                                                                                                ▒
//        │    │  data16   data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)                                                                                              ▒
//        │100:│  mov      %r14,-0x68(%rbp)                                                                                                                                  ▒
//        │    │  test     %rbx,%rbx                                                                                                                                         ▒
//        │    │↓ jle      1cb                                                                                                                                               ▒
//        │    │  dec      %rbx                                                                                                                                              ▒
//        │    └──jne      b0         
// RandomBitwiseBranch
//   0.68 │ c0:   mov       -0x4(%r13,%rax,1),%ecx                                                                                                                           ▒
//   0.51 │       mov       0x0(%r13,%rax,1),%edx                                                                                                                            ▒
//   3.25 │       or        -0x4(%r12,%rax,1),%ecx                                                                                                                           ▒
//   4.06 │       or        (%r12,%rax,1),%edx                                                                                                                               ▒
//   3.30 │       movd      %ecx,%xmm1                                                                                                                                       ▒
//   9.48 │       movd      %edx,%xmm2                                                                                                                                       ▒
//   8.89 │       punpcklbw %xmm1,%xmm1                                                                                                                                      ▒
//   6.00 │       punpcklwd %xmm1,%xmm1  
// ...
//   0.85 │    │  pand      %xmm1,%xmm5                                                                                                                                      ▒
//   1.29 │    │  pandn     %xmm7,%xmm1                                                                                                                                      ▒
//   1.74 │    │  por       %xmm5,%xmm1                                                                                                                                      ▒
//   3.07 │    │  pand      %xmm2,%xmm6                                                                                                                                      ▒
//   3.24 │    │  pandn     %xmm8,%xmm2                                                                                                                                      ▒
//   3.67 │    │  por       %xmm6,%xmm2                                                                                                                                      ▒
//   2.23 │    │  movdqu    %xmm1,-0x10(%r14,%rax,4)                                                                                                                         ▒
//   0.91 │    │  movdqu    %xmm2,(%r14,%rax,4)                                                                                                                              ▒
//   1.81 │    │  add       $0x8,%rax                                                                                                                                        ▒
//   1.04 │    ├──cmp       $0x100004,%rax                                                                                                                                   ▒
//        │    └──jne       c0        
BENCHMARK_MAIN();