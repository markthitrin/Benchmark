#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <random>
#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void RandomAccess(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  volatile __m256i* const p0 = static_cast<__m256i*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(__m256i);
  __m256i sink0; memset(&sink0, 0x1b, sizeof(sink0));
  __m256i sink = sink0;

  int* ind0 = (int*)malloc(sizeof(int) * array_size);
  for(int i = 0;i < array_size;i++) {
    ind0[i] = i;
  }
  std::shuffle(ind0, ind0 + array_size, std::default_random_engine(0));

  for(auto _ : state) {
    const int* ind = ind0;
    while(ind != ind0 + array_size) {
      REPEAT(sink = *(p0 + *ind++););
    }
  }
  benchmark::DoNotOptimize(sink);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size);
  free(memory);
}

// range(0) is memory size in byte
BENCHMARK(RandomAccess)->RangeMultiplier(2)->Range(8192, 1<<30);

//g++
// ----------------------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------
// RandomAccess/8192              249 ns          249 ns      2808338 bytes_per_second=30.6085Gi/s items_per_second=4.01192M/s
// RandomAccess/16384             502 ns          502 ns      1387683 bytes_per_second=30.4198Gi/s items_per_second=1.99359M/s
// RandomAccess/32768            1000 ns          999 ns       696284 bytes_per_second=30.5352Gi/s items_per_second=1.00058M/s L1 1 instance
// RandomAccess/65536            2005 ns         2004 ns       349510 bytes_per_second=30.4502Gi/s items_per_second=498.895k/s
// RandomAccess/131072           4049 ns         4048 ns       174634 bytes_per_second=30.1567Gi/s items_per_second=247.043k/s
// RandomAccess/262144           8136 ns         8133 ns        86122 bytes_per_second=30.0178Gi/s items_per_second=122.953k/s L1 MAX
// RandomAccess/524288          16447 ns        16445 ns        42480 bytes_per_second=29.6922Gi/s items_per_second=60.8096k/s L2 1 instance
// RandomAccess/1048576         33272 ns        33265 ns        21092 bytes_per_second=29.3573Gi/s items_per_second=30.0619k/s
// RandomAccess/2097152         68366 ns        68353 ns        10328 bytes_per_second=28.574Gi/s items_per_second=14.6299k/s  L3 1 instance
// RandomAccess/4194304        137349 ns       137275 ns         5096 bytes_per_second=28.4556Gi/s items_per_second=7.28464k/s L2 MAX
// RandomAccess/8388608        294179 ns       294054 ns         2405 bytes_per_second=26.5682Gi/s items_per_second=3.40073k/s L3 MAX
// RandomAccess/16777216       946114 ns       945865 ns          742 bytes_per_second=16.5193Gi/s items_per_second=1.05723k/s
// RandomAccess/33554432      2838852 ns      2838236 ns          245 bytes_per_second=11.0104Gi/s items_per_second=352.331/s
// RandomAccess/67108864      7742033 ns      7739718 ns           85 bytes_per_second=8.07523Gi/s items_per_second=129.204/s
// RandomAccess/134217728    18152243 ns     18150597 ns           35 bytes_per_second=6.88683Gi/s items_per_second=55.0946/s
// RandomAccess/268435456    46486194 ns     46480330 ns           15 bytes_per_second=5.37862Gi/s items_per_second=21.5145/s
// RandomAccess/536870912   133940604 ns    133931380 ns            5 bytes_per_second=3.73326Gi/s items_per_second=7.46651/s
// RandomAccess/1073741824  355225133 ns    355155990 ns            2 bytes_per_second=2.81566Gi/s items_per_second=2.81566/s
//   0.39 │       shl      $0x5,%rdx                                                                                                                                         ▒
//   0.32 │       movaps   %xmm0,0x90(%rsp)                                                                                                                                  ▒
//   0.25 │       add      %r12,%rdx                                                                                                                                         ▒
//   0.43 │       movaps   %xmm0,0xb0(%rsp)                                                                                                                                  ▒
//   0.67 │       movdqa   (%rdx),%xmm1                                                                                                                                      ▒
//   0.49 │       movaps   %xmm1,0xc0(%rsp)                                                                                                                                  ▒
//   0.21 │       movdqa   0x10(%rdx),%xmm0                                                                                                                                  ▒
//   0.22 │       movslq   0x8(%rax),%rdx                                                                                                                                    ▒
//   0.33 │       movaps   %xmm1,0x80(%rsp)                                                                                                                                  ▒
//   0.30 │       shl      $0x5,%rdx                                                                                                                                         ▒
//   0.20 │       movaps   %xmm0,0x90(%rsp)                                                                                                                                  ▒
//   0.30 │       add      %r12,%rdx                                                                                                                                         ◆
//   0.47 │       movaps   %xmm0,0xd0(%rsp)                                                                                                                                  ▒
//   0.44 │       movdqa   (%rdx),%xmm1                                                                                                                                      ▒
//   0.33 │       movaps   %xmm1,0xe0(%rsp)                                                                                                                                  ▒
//   0.34 │       movdqa   0x10(%rdx),%xmm0                                                                                                                                  ▒
//   0.42 │       movslq   0xc(%rax),%rdx                                                                                                                                    ▒
//   0.31 │       movaps   %xmm1,0x80(%rsp) 


// ----------------------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------
// RandomAccess/8192              105 ns          105 ns      6620334 bytes_per_second=72.3381Gi/s items_per_second=9.4815M/s
// RandomAccess/16384             256 ns          256 ns      2728076 bytes_per_second=59.5126Gi/s items_per_second=3.90022M/s
// RandomAccess/32768             557 ns          557 ns      1243083 bytes_per_second=54.765Gi/s items_per_second=1.79454M/s  L1 1 instance
// RandomAccess/65536            1158 ns         1157 ns       603598 bytes_per_second=52.7373Gi/s items_per_second=864.048k/s
// RandomAccess/131072           2354 ns         2354 ns       298793 bytes_per_second=51.8581Gi/s items_per_second=424.822k/s
// RandomAccess/262144           4887 ns         4887 ns       145640 bytes_per_second=49.96Gi/s items_per_second=204.636k/s   L1 MAX
// RandomAccess/524288          11183 ns        11182 ns        62338 bytes_per_second=43.6683Gi/s items_per_second=89.4326k/s L2 1 instance
// RandomAccess/1048576         24015 ns        24010 ns        29153 bytes_per_second=40.6723Gi/s items_per_second=41.6485k/s
// RandomAccess/2097152         49531 ns        49519 ns        13965 bytes_per_second=39.4416Gi/s items_per_second=20.1941k/s L3 1 instance
// RandomAccess/4194304        100976 ns       100971 ns         6886 bytes_per_second=38.6869Gi/s items_per_second=9.90385k/s L2 MAX
// RandomAccess/8388608        205446 ns       205436 ns         3391 bytes_per_second=38.0288Gi/s items_per_second=4.86769k/s L3 MAX
// RandomAccess/16777216       831321 ns       831265 ns          845 bytes_per_second=18.7967Gi/s items_per_second=1.20299k/s
// RandomAccess/33554432      2846825 ns      2846514 ns          243 bytes_per_second=10.9783Gi/s items_per_second=351.307/s
// RandomAccess/67108864      7115275 ns      7114487 ns           87 bytes_per_second=8.78489Gi/s items_per_second=140.558/s
// RandomAccess/134217728    16095686 ns     16094366 ns           39 bytes_per_second=7.76669Gi/s items_per_second=62.1335/s
// RandomAccess/268435456    39769014 ns     39765019 ns           17 bytes_per_second=6.28693Gi/s items_per_second=25.1477/s
// RandomAccess/536870912   111316711 ns    111285581 ns            6 bytes_per_second=4.49295Gi/s items_per_second=8.98589/s
// RandomAccess/1073741824  312396113 ns    312317273 ns            2 bytes_per_second=3.20187Gi/s items_per_second=3.20187/s
//   0.58 │       shl      $0x5,%rcx                                                                                                                                         ▒
//   0.68 │       movaps   (%r12,%rcx,1),%xmm0                                                                                                                               ▒
//   0.65 │       movaps   0x10(%r12,%rcx,1),%xmm0                                                                                                                           ▒
//   0.68 │       movslq   0x4(%rax),%rcx                                                                                                                                    ▒
//   0.73 │       shl      $0x5,%rcx                                                                                                                                         ▒
//   0.78 │       movaps   (%r12,%rcx,1),%xmm0                                                                                                                               ▒
//   0.79 │       movaps   0x10(%r12,%rcx,1),%xmm0                                                                                                                           ▒
//   0.79 │       movslq   0x8(%rax),%rcx   



// Note1 : L1d 256 KB (8 instances)   32  KB each
//         L2    4 MB (8 instances)   512 KB each
//         L3    8 MB (2 instances)   2   MB each
BENCHMARK_MAIN();