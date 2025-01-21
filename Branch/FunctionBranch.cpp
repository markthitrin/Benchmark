#include <benchmark/benchmark.h>
#include <stdlib.h>

constexpr int array_size = (1 << 20);

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

int add(const int& a,const int& b) {
    return a + b;
}

int mul(const int& a,const int& b) {
    return a * b;
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
        a[i] = add(a[i],b[i]);
      }
      else {
        a[i] = mul(a[i],b[i]);
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
        a[i] = add(a[i],b[i]);
      }
      else {
        a[i] = mul(a[i],b[i]);
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
      a[i] = c[i] ? add(a[i],b[i]) : mul(a[i],b[i]);
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
      a[i] = c[i] ? add(a[i],b[i]) : mul(a[i],b[i]);
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
    int (*func[2])(const int&,const int&) = {&add, &mul};
    for(int i = 0;i < array_size;i++) {
      a[i] = func[c[i]](a[i],b[i]);
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
    int (*func[2])(const int&,const int&) = {&add, &mul};
    for(int i = 0;i < array_size;i++) {
        a[i] = func[c[i]](a[i],b[i]);
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
// PredictableLogicalBranch      1082973 ns      1082964 ns          646 items_per_second=923.391/s
// UnpredictableLogicalBranch    3669991 ns      3669804 ns          191 items_per_second=272.494/s
// PredictableTernaryBranch      1102057 ns      1101954 ns          593 items_per_second=907.479/s
// UnpredictableTernaryBranch    3671289 ns      3671288 ns          191 items_per_second=272.384/s
// PredictableIndexBranch        1571139 ns      1570712 ns          445 items_per_second=636.654/s
// UnpredictableIndexBranch      5145711 ns      5145773 ns          136 items_per_second=194.334/s
// PredictableLogicalBranch
//  10.58 │160:┌─→add      %ecx,%edx                                                                                                                                         ▒
//   9.67 │    │  mov      %edx,(%rbx,%rax,4)                                                                                                                                ▒
//  11.14 │    │  add      $0x1,%rax                                                                                                                                         ▒
//  15.14 │    │  cmp      $0x100000,%rax                                                                                                                                    ▒
//        │    │↓ je       192                                                                                                                                               ▒
//  10.73 │171:│  cmpb     $0x0,(%r14,%rax,1)                                                                                                                                ▒
//  11.24 │    │  mov      (%rbx,%rax,4),%ecx                                                                                                                                ▒
//  14.34 │    │  mov      0x0(%r13,%rax,4),%edx                                                                                                                             ◆
//  13.96 │    └──jne      160
// // UnpredicatbleIndexBranch
//   3.18 │1a0:┌─→movzbl     (%r14),%eax                                                                                                                                     ▒
//   3.18 │    │  mov        %r15,%rsi                                                                                                                                       ▒
//   5.90 │    │  mov        %r13,%rdi                                                                                                                                       ▒
//   3.44 │    │  add        $0x4,%r13                                                                                                                                       ▒
//   8.75 │    │  add        $0x1,%r14                                                                                                                                       ▒
//   9.38 │    │  add        $0x4,%r15                                                                                                                                       ▒
//  56.82 │    │→ call       *-0x60(%rbp,%rax,8)                                                                                                                             ▒
//   3.46 │    │  mov        %eax,-0x4(%r13)                                                                                                                                 ▒
//   2.69 │    ├──cmp        %rbx,%r13                                                                                                                                       ▒
//        │    └──jne        1a0    


// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// PredictableLogicalBranch      1039398 ns      1039224 ns          664 items_per_second=962.257/s
// UnpredictableLogicalBranch    1007653 ns      1007421 ns          698 items_per_second=992.634/s
// PredictableTernaryBranch       990591 ns       990234 ns          708 items_per_second=1.00986k/s
// UnpredictableTernaryBranch    1053305 ns      1053131 ns          699 items_per_second=949.549/s
// PredictableIndexBranch        1822895 ns      1822351 ns          385 items_per_second=548.742/s
// UnpredictableIndexBranch      5168465 ns      5167489 ns          135 items_per_second=193.518/s
// PredictableIndexBranch
//  b0:┌─→movzbl   (%r12,%rbx,1),%eax                                                                                                                                ▒
//   9.73 │    │  mov      %r13,%rdi                                                                                                                                         ▒
//   5.73 │    │  mov      %r15,%rsi                                                                                                                                         ◆
//  48.76 │    │→ call     *(%r14,%rax,8)                                                                                                                                    ▒
//   2.94 │    │  mov      %eax,0x0(%r13)                                                                                                                                    ▒
//   3.17 │    │  inc      %rbx                                                                                                                                              ▒
//   5.96 │    │  add      $0x4,%r15                                                                                                                                         ▒
//   3.10 │    │  add      $0x4,%r13                                                                                                                                         ▒
//   5.00 │    ├──cmp      $0x100000,%rbx                                                                                                                                    ▒
//        │    └──jne      b0       
BENCHMARK_MAIN();