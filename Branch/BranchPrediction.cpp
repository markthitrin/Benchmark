#include <benchmark/benchmark.h>
#include <stdlib.h>

constexpr int array_size = (1 << 20);

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void PredictableLogicalBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() > 0;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      if(c[i]) {
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

static void UnpredictableLogicalBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      if(c[i]) {
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

static void PredictableTernaryBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() > 0;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      a[i] = c[i] ? a[i] + b[i] : a[i] * b[i];
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

static void UnpredictableTernaryBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      a[i] = c[i] ? a[i] + b[i] : a[i] * b[i];
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

static void PredictableIndexBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() > 0;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      int arr[] = {a[i] + b[i],a[i] * b[i]};
      a[i] = arr[c[i]];
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

static void UnpredictableIndexBranch(benchmark::State& state) {
  int* a = new int[array_size];
  int* b = new int[array_size];
  bool* c = new bool[array_size];
  for(int i = 0;i < array_size;i++) {
    a[i] = std::rand();
    b[i] = std::rand();
    c[i] = std::rand() & 1;
  }
  for(auto _ : state) {
    for(int i = 0;i < array_size;i++) {
      int arr[] = {a[i] + b[i],a[i] * b[i]};
      a[i] = arr[c[i]];
    }
    escape(a);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(PredictableLogicalBranch);
BENCHMARK(UnpredictableLogicalBranch);
BENCHMARK(PredictableTernaryBranch);
BENCHMARK(UnpredictableTernaryBranch);
BENCHMARK(PredictableIndexBranch);
BENCHMARK(UnpredictableIndexBranch);

// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// PredictableLogicalBranch      1057038 ns      1056284 ns          605 items_per_second=946.715/s
// UnpredictableLogicalBranch    3794191 ns      3791685 ns          186 items_per_second=263.735/s
// PredictableTernaryBranch      1033877 ns      1032983 ns          647 items_per_second=968.07/s
// UnpredictableTernaryBranch    3784133 ns      3781737 ns          186 items_per_second=264.429/s
// PredictableIndexBranch        1150357 ns      1149642 ns          611 items_per_second=869.836/s
// UnpredictableIndexBranch      1142476 ns      1141606 ns          615 items_per_second=875.959/s

// UnpredictableLogicalBranch
//  13.93 │160:   add      %ecx,%edx                                                                                                                                         ▒
//  14.11 │       mov      %edx,(%rbx,%rax,4)                                                                                                                                ▒
//  14.00 │       add      $0x1,%rax                                                                                                                                         ▒
//   1.75 │       cmp      $0x100000,%rax                                                                                                                                    ▒
//        │     ↓ je       192                                                                                                                                               ▒
//   3.67 │171:┌─→cmpb     $0x0,0x0(%r13,%rax,1)                                                                                                                             ▒
//   6.62 │    │  mov      (%rbx,%rax,4),%ecx                                                                                                                                ▒
//   7.15 │    │  mov      (%r14,%rax,4),%edx                                                                                                                                ▒
//   4.54 │    │↑ jne      160                                                                                                                                               ▒
//   6.43 │    │  imul     %ecx,%edx                                                                                                                                         ▒
//  11.46 │    │  mov      %edx,(%rbx,%rax,4)                                                                                                                                ▒
//  10.15 │    │  add      $0x1,%rax                                                                                                                                         ▒
//   5.37 │    ├──cmp      $0x100000,%rax                                                                                                                                    ▒
//        │    └──jne      171            
// UnpredictableTernaryBranch
//  13.93 │160:   add      %ecx,%edx                                                                                                                                         ▒
//  14.11 │       mov      %edx,(%rbx,%rax,4)                                                                                                                                ▒
//  14.00 │       add      $0x1,%rax                                                                                                                                         ▒
//   1.75 │       cmp      $0x100000,%rax                                                                                                                                    ▒
//        │     ↓ je       192                                                                                                                                               ▒
//   3.67 │171:┌─→cmpb     $0x0,0x0(%r13,%rax,1)                                                                                                                             ▒
//   6.62 │    │  mov      (%rbx,%rax,4),%ecx                                                                                                                                ▒
//   7.15 │    │  mov      (%r14,%rax,4),%edx                                                                                                                                ▒
//   4.54 │    │↑ jne      160                                                                                                                                               ▒
//   6.43 │    │  imul     %ecx,%edx                                                                                                                                         ▒
//  11.46 │    │  mov      %edx,(%rbx,%rax,4)                                                                                                                                ▒
//  10.15 │    │  add      $0x1,%rax                                                                                                                                         ◆
//   5.37 │    ├──cmp      $0x100000,%rax                                                                                                                                    ▒
//        │    └──jne      171      
// UnpredicatbleIndexBranch
//   8.03 │158:┌─→mov      (%r12,%rax,4),%edx                                                                                                                                ▒
//   8.03 │    │  mov      (%r14,%rax,4),%ecx                                                                                                                                ▒
//  11.18 │    │  lea      (%rdx,%rcx,1),%esi                                                                                                                                ▒
//  10.85 │    │  imul     %ecx,%edx                                                                                                                                         ▒
//  10.96 │    │  mov      %esi,-0x60(%rbp)                                                                                                                                  ▒
//  11.44 │    │  mov      %edx,-0x5c(%rbp)                                                                                                                                  ▒
//   8.49 │    │  movzbl   0x0(%r13,%rax,1),%edx                                                                                                                             ▒
//   5.69 │    │  mov      -0x60(%rbp,%rdx,4),%edx                                                                                                                           ▒
//   8.39 │    │  mov      %edx,(%r12,%rax,4)                                                                                                                                ▒
//   7.97 │    │  add      $0x1,%rax                                                                                                                                         ▒
//   8.03 │    ├──cmp      $0x100000,%rax                                                                                                                                    ▒
//        │    └──jne      158      


// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// PredictableLogicalBranch      1116814 ns      1119562 ns          628 items_per_second=893.207/s
// UnpredictableLogicalBranch    1054350 ns      1056901 ns          676 items_per_second=946.162/s
// PredictableTernaryBranch       971729 ns       973720 ns          671 items_per_second=1.02699k/s
// UnpredictableTernaryBranch    1004745 ns      1006644 ns          820 items_per_second=993.4/s
// PredictableIndexBranch        1139987 ns      1141758 ns          608 items_per_second=875.842/s
// UnpredictableIndexBranch      1151689 ns      1153563 ns          613 items_per_second=866.879/s

// UnpredicatbleLogicalBranch, UnpredictableTernaryBranch
//  3.01 │ a0:┌─→movd      -0x4(%r12,%rax,1),%xmm0                                                                                                                          ▒
//   2.56 │    │  punpcklbw %xmm0,%xmm0                                                                                                                                      ▒
//   5.83 │    │  punpcklwd %xmm0,%xmm0                                                                                                                                      ▒
//   2.97 │    │  movd      (%r12,%rax,1),%xmm1                                                                                                                              ▒
//   0.56 │    │  punpcklbw %xmm1,%xmm1                                                                                                                                      ▒
//   1.06 │    │  punpcklwd %xmm1,%xmm1                                                                                                                                      ▒
//   1.21 │    │  pslld     $0x1f,%xmm0                                                                                                                                      ▒
//   0.65 │    │  psrad     $0x1f,%xmm0                                                                                                                                      ▒
//   0.50 │    │  pslld     $0x1f,%xmm1                                                                                                                                      ▒
//   0.79 │    │  psrad     $0x1f,%xmm1                                                                                                                                      ▒
//   0.91 │    │  movdqu    -0x10(%r14,%rax,4),%xmm3                                                                                                                         ▒
//   0.38 │    │  movdqu    (%r14,%rax,4),%xmm4                                                                                                                              ▒
//   0.62 │    │  movdqu    -0x10(%r15,%rax,4),%xmm5                                                                                                                         ▒
//   1.00 │    │  movdqu    (%r15,%rax,4),%xmm6                                                                                                                              ▒
//   1.15 │    │  movdqa    %xmm5,%xmm7                                                                                                                                      ▒
//   0.50 │    │  paddd     %xmm3,%xmm7                                                                                                                                      ▒
//   0.53 │    │  movdqa    %xmm6,%xmm2                                                                                                                                      ▒
//   0.71 │    │  paddd     %xmm4,%xmm2                                                                                                                                      ▒
//   1.24 │    │  pshufd    $0xf5,%xmm5,%xmm8                                                                                                                                ▒
//   0.53 │    │  pmuludq   %xmm3,%xmm5                                                                                                                                      ▒
//   0.56 │    │  pshufd    $0xe8,%xmm5,%xmm5                                                                                                                                ▒
//   1.09 │    │  pshufd    $0xf5,%xmm3,%xmm3                                                                                                                                ▒
//   1.29 │    │  pmuludq   %xmm8,%xmm3                                                                                                                                      ◆
//   0.68 │    │  pshufd    $0xe8,%xmm3,%xmm3                                                                                                                                ▒
//   0.59 │    │  punpckldq %xmm3,%xmm5                                                                                                                                      ▒
//   1.21 │    │  pshufd    $0xf5,%xmm6,%xmm3                                                                                                                                ▒
//   1.23 │    │  pmuludq   %xmm4,%xmm6                                                                                                                                      ▒
//   0.44 │    │  pshufd    $0xe8,%xmm6,%xmm6                                                                                                                                ▒
//   0.94 │    │  pshufd    $0xf5,%xmm4,%xmm4                                                                                                                                ▒
//   2.18 │    │  pmuludq   %xmm3,%xmm4                                                                                                                                      ▒
//   2.42 │    │  pshufd    $0xe8,%xmm4,%xmm3                                                                                                                                ▒
//   1.03 │    │  punpckldq %xmm3,%xmm6                                                                                                                                      ▒
//   1.92 │    │  pand      %xmm0,%xmm7                                                                                                                                      ▒
//   4.26 │    │  pandn     %xmm5,%xmm0                                                                                                                                      ▒
//   4.71 │    │  por       %xmm7,%xmm0                                                                                                                                      ▒
//   2.23 │    │  pand      %xmm1,%xmm2                                                                                                                                      ▒
//   6.56 │    │  pandn     %xmm6,%xmm1                                                                                                                                      ▒
//   6.80 │    │  por       %xmm2,%xmm1                                                                                                                                      ▒
//   6.39 │    │  movdqu    %xmm0,-0x10(%r14,%rax,4)                                                                                                                         ▒
//  12.83 │    │  movdqu    %xmm1,(%r14,%rax,4)                                                                                                                              ▒
//   5.89 │    │  add       $0x8,%rax                                                                                                                                        ▒
//   3.30 │    ├──cmp       $0x100004,%rax                                                                                                                                   ▒
//        │    └──jne       a0     
// UnpredictableIndexBranch
//   1.86 │ a0:┌─→mov      -0x4(%r14,%rax,4),%ecx                                                                                                                            ▒
//   1.95 │    │  mov      -0x4(%r15,%rax,4),%edx                                                                                                                            ▒
//   4.04 │    │  lea      (%rdx,%rcx,1),%esi                                                                                                                                ▒
//   2.18 │    │  mov      %esi,-0x48(%rbp)                                                                                                                                  ▒
//   6.84 │    │  imul     %ecx,%edx                                                                                                                                         ▒
//   6.79 │    │  mov      %edx,-0x44(%rbp)                                                                                                                                  ▒
//   6.28 │    │  movzbl   -0x1(%r12,%rax,1),%ecx                                                                                                                            ▒
//  13.15 │    │  mov      -0x48(%rbp,%rcx,4),%ecx                                                                                                                           ▒
//   6.25 │    │  mov      %ecx,-0x4(%r14,%rax,4)                                                                                                                            ◆
//   3.45 │    │  mov      (%r14,%rax,4),%ecx                                                                                                                                ▒
//   3.66 │    │  mov      (%r15,%rax,4),%edx                                                                                                                                ▒
//   3.98 │    │  lea      (%rdx,%rcx,1),%esi                                                                                                                                ▒
//   7.82 │    │  mov      %esi,-0x48(%rbp)                                                                                                                                  ▒
//   4.49 │    │  imul     %ecx,%edx                                                                                                                                         ▒
//   3.78 │    │  mov      %edx,-0x44(%rbp)                                                                                                                                  ▒
//   3.57 │    │  movzbl   (%r12,%rax,1),%ecx                                                                                                                                ▒
//   3.98 │    │  mov      -0x48(%rbp,%rcx,4),%ecx                                                                                                                           ▒
//   6.29 │    │  mov      %ecx,(%r14,%rax,4)                                                                                                                                ▒
//   3.07 │    │  add      $0x2,%rax                                                                                                                                         ▒
//   1.74 │    ├──cmp      $0x100001,%rax                                                                                                                                    ▒
//        │    └──jne      a0   
BENCHMARK_MAIN();