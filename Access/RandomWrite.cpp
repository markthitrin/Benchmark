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

static void RandomWrite(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  volatile __m256i* const p0 = static_cast<__m256i*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(__m256i);
  __m256i fill0; memset(&fill0, 0x1b, sizeof(fill0));
  __m256i fill = fill0;

  int* ind0 = (int*)malloc(sizeof(int) * array_size);
  for(int i = 0;i < array_size;i++) {
    ind0[i] = i;
  }
  std::shuffle(ind0, ind0 + array_size, std::default_random_engine(0));

  for(auto _ : state) {
    const int* ind = ind0;
    while(ind != ind0 + array_size) {
      REPEAT(*(p0 + *ind++) = fill;)
    }
  }
  benchmark::DoNotOptimize(fill);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size);
  free(memory);
}

// range(0) is memory size in byte
BENCHMARK(RandomWrite)->RangeMultiplier(2)->Range(1024, 1<<30);

// g++
// ---------------------------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------
// RandomWrite/1024             10.1 ns         10.1 ns     66786535 bytes_per_second=94.7234Gi/s items_per_second=99.3247M/s
// RandomWrite/2048             19.1 ns         19.1 ns     36655815 bytes_per_second=100.015Gi/s items_per_second=52.4366M/s
// RandomWrite/4096             38.3 ns         38.3 ns     18080304 bytes_per_second=99.7296Gi/s items_per_second=26.1435M/s
// RandomWrite/8192             76.0 ns         76.0 ns      9178037 bytes_per_second=100.414Gi/s items_per_second=13.1614M/s
// RandomWrite/16384             154 ns          154 ns      4521003 bytes_per_second=99.1688Gi/s items_per_second=6.49913M/s
// RandomWrite/32768             413 ns          413 ns      1690449 bytes_per_second=73.8709Gi/s items_per_second=2.4206M/s
// RandomWrite/65536            1547 ns         1548 ns       449773 bytes_per_second=39.4404Gi/s items_per_second=646.192k/s
// RandomWrite/131072           3340 ns         3341 ns       205956 bytes_per_second=36.5419Gi/s items_per_second=299.351k/s
// RandomWrite/262144           7137 ns         7137 ns        97765 bytes_per_second=34.2076Gi/s items_per_second=140.114k/s
// RandomWrite/524288          15754 ns        15756 ns        44187 bytes_per_second=30.9901Gi/s items_per_second=63.4677k/s
// RandomWrite/1048576         35935 ns        35938 ns        19366 bytes_per_second=27.1738Gi/s items_per_second=27.826k/s
// RandomWrite/2097152         80088 ns        80100 ns         7977 bytes_per_second=24.3836Gi/s items_per_second=12.4844k/s
// RandomWrite/4194304        663214 ns       663261 ns          958 bytes_per_second=5.88946Gi/s items_per_second=1.5077k/s
// RandomWrite/8388608       3745727 ns      3745841 ns          187 bytes_per_second=2.08565Gi/s items_per_second=266.963/s
// RandomWrite/16777216      8800155 ns      8798847 ns           72 bytes_per_second=1.7758Gi/s items_per_second=113.651/s
// RandomWrite/33554432     19200921 ns     19199646 ns           34 bytes_per_second=1.62763Gi/s items_per_second=52.0843/s
// RandomWrite/67108864     40798720 ns     40787885 ns           17 bytes_per_second=1.53232Gi/s items_per_second=24.5171/s
// RandomWrite/134217728    86860729 ns     86862792 ns            8 bytes_per_second=1.43905Gi/s items_per_second=11.5124/s
// RandomWrite/268435456   202383778 ns    202370687 ns            3 bytes_per_second=1.23536Gi/s items_per_second=4.94143/s
// RandomWrite/536870912   594414207 ns    594461926 ns            1 bytes_per_second=861.283Mi/s items_per_second=1.68219/s
// RandomWrite/1073741824 1658477810 ns   1644296017 ns            1 bytes_per_second=622.759Mi/s items_per_second=0.608163/s
//   0.71 │       shl          $0x5,%rdx                                                                                                                                     ▒
//   0.73 │       add          %r12,%rdx                                                                                                                                     ▒
//   0.84 │       vmovdqa      %ymm0,(%rdx)                                                                                                                                  ▒
//   0.83 │       movslq       0x14(%rax),%rdx                                                                                                                               ▒
//   0.65 │       shl          $0x5,%rdx                                                                                                                                     ▒
//   0.81 │       add          %r12,%rdx                                                                                                                                     ▒
//   0.73 │       vmovdqa      %ymm0,(%rdx)                                                                                                                                  ▒
//   0.62 │       movslq       0x18(%rax),%rdx                                                                                                                               ◆
//   0.65 │       vmovdqa      0x80(%rsp),%ymm0 


// clang++
// ---------------------------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------
// RandomWrite/1024             10.7 ns         10.6 ns     65061180 bytes_per_second=89.5572Gi/s items_per_second=93.9075M/s
// RandomWrite/2048             20.8 ns         20.8 ns     33750913 bytes_per_second=91.8018Gi/s items_per_second=48.1306M/s
// RandomWrite/4096             42.1 ns         42.1 ns     16664321 bytes_per_second=90.6146Gi/s items_per_second=23.7541M/s
// RandomWrite/8192             84.4 ns         84.3 ns      8270316 bytes_per_second=90.4925Gi/s items_per_second=11.861M/s
// RandomWrite/16384             172 ns          171 ns      4085796 bytes_per_second=89.0225Gi/s items_per_second=5.83418M/s
// RandomWrite/32768             364 ns          363 ns      1848220 bytes_per_second=83.9835Gi/s items_per_second=2.75197M/s
// RandomWrite/65536            1185 ns         1183 ns       597783 bytes_per_second=51.6017Gi/s items_per_second=845.443k/s
// RandomWrite/131072           2702 ns         2698 ns       261825 bytes_per_second=45.2489Gi/s items_per_second=370.679k/s
// RandomWrite/262144           5677 ns         5667 ns       116689 bytes_per_second=43.0775Gi/s items_per_second=176.445k/s
// RandomWrite/524288          13117 ns        13096 ns        51372 bytes_per_second=37.2838Gi/s items_per_second=76.3573k/s
// RandomWrite/1048576         29807 ns        29764 ns        24213 bytes_per_second=32.8102Gi/s items_per_second=33.5976k/s
// RandomWrite/2097152         61722 ns        61622 ns        10116 bytes_per_second=31.695Gi/s items_per_second=16.2279k/s
// RandomWrite/4194304       1397286 ns      1395082 ns          888 bytes_per_second=2.80001Gi/s items_per_second=716.804/s
// RandomWrite/8388608       6328333 ns      6319166 ns          178 bytes_per_second=1.23632Gi/s items_per_second=158.249/s
// RandomWrite/16777216     11463268 ns     11445680 ns           49 bytes_per_second=1.36514Gi/s items_per_second=87.3692/s
// RandomWrite/33554432     32683099 ns     32636659 ns           34 bytes_per_second=980.493Mi/s items_per_second=30.6404/s
// RandomWrite/67108864     62501869 ns     62411354 ns           15 bytes_per_second=1.00142Gi/s items_per_second=16.0227/s
// RandomWrite/134217728   178251528 ns    178005500 ns            5 bytes_per_second=719.079Mi/s items_per_second=5.6178/s
// RandomWrite/268435456   397393141 ns    396854338 ns            2 bytes_per_second=645.073Mi/s items_per_second=2.51982/s
// RandomWrite/536870912  1162950985 ns   1161449941 ns            1 bytes_per_second=440.828Mi/s items_per_second=0.860993/s
// RandomWrite/1073741824 1445263140 ns   1443281509 ns            1 bytes_per_second=709.494Mi/s items_per_second=0.692866/s
//   0.72 │       shl          $0x5,%rcx                                                                                                                                     ▒
//   1.15 │       vmovapd      %ymm0,(%r12,%rcx,1)                                                                                                                           ▒
//   1.10 │       movslq       0x34(%rax),%rcx                                                                                                                               ▒
//   1.08 │       shl          $0x5,%rcx                                                                                                                                     ◆
//   1.98 │       vmovapd      %ymm0,(%r12,%rcx,1)                                                                                                                           ▒
//   1.07 │       movslq       0x38(%rax),%rcx                   

// The result in slower bandwith in clang++ is still unknown.

BENCHMARK_MAIN();