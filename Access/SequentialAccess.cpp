#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <cstring>
#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void SequentialAccess(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  volatile __m256i* const p0 = static_cast<__m256i*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(__m256i);
  __m256i sink0; memset(&sink0, 0x1b, sizeof(sink0));
  __m256i sink = sink0;
  int u = 0;

  for(auto _ : state) {
    volatile __m256i* p = p0;
    while(p != end) {
      REPEAT(sink = *p++;)
    }
  }
  benchmark::DoNotOptimize(sink);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size);
  free(memory);
}

// range(0) is memory size in byte
BENCHMARK(SequentialAccess)->RangeMultiplier(2)->Range(1024, 1 << 30);

// g++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// SequentialAccess/1024             3.81 ns         3.81 ns    182676373 bytes_per_second=250.291Gi/s items_per_second=262.449M/s
// SequentialAccess/2048             7.64 ns         7.64 ns     91610851 bytes_per_second=249.616Gi/s items_per_second=130.871M/s
// SequentialAccess/4096             15.3 ns         15.3 ns     45477869 bytes_per_second=248.87Gi/s items_per_second=65.2397M/s
// SequentialAccess/8192             30.6 ns         30.6 ns     22889060 bytes_per_second=249.698Gi/s items_per_second=32.7285M/s
// SequentialAccess/16384            85.3 ns         85.2 ns      8168485 bytes_per_second=179.037Gi/s items_per_second=11.7334M/s
// SequentialAccess/32768             189 ns          189 ns      3703915 bytes_per_second=161.619Gi/s items_per_second=5.29592M/s L1 1 instance
// SequentialAccess/65536             414 ns          414 ns      1689916 bytes_per_second=147.364Gi/s items_per_second=2.41442M/s
// SequentialAccess/131072            920 ns          920 ns       761529 bytes_per_second=132.672Gi/s items_per_second=1.08685M/s
// SequentialAccess/262144           1935 ns         1935 ns       364590 bytes_per_second=126.201Gi/s items_per_second=516.917k/s L1 MAX
// SequentialAccess/524288           3893 ns         3892 ns       179765 bytes_per_second=125.444Gi/s items_per_second=256.909k/s L2 1 instance
// SequentialAccess/1048576          7798 ns         7796 ns        89136 bytes_per_second=125.267Gi/s items_per_second=128.274k/s
// SequentialAccess/2097152         15616 ns        15613 ns        44647 bytes_per_second=125.099Gi/s items_per_second=64.0508k/s L3 1 instance
// SequentialAccess/4194304         31190 ns        31184 ns        22414 bytes_per_second=125.266Gi/s items_per_second=32.068k/s  L2 MAX
// SequentialAccess/8388608         62650 ns        62637 ns        11150 bytes_per_second=124.726Gi/s items_per_second=15.9649k/s L3 MAX
// SequentialAccess/16777216       127164 ns       127126 ns         5523 bytes_per_second=122.909Gi/s items_per_second=7.86619k/s
// SequentialAccess/33554432       255121 ns       255076 ns         2704 bytes_per_second=122.513Gi/s items_per_second=3.9204k/s
// SequentialAccess/67108864       518073 ns       517959 ns         1129 bytes_per_second=120.666Gi/s items_per_second=1.93066k/s
// SequentialAccess/134217728     1056323 ns      1056046 ns          564 bytes_per_second=118.366Gi/s items_per_second=946.928/s
// SequentialAccess/268435456     2175951 ns      2175488 ns          287 bytes_per_second=114.917Gi/s items_per_second=459.667/s
// SequentialAccess/536870912     4823730 ns      4822821 ns          114 bytes_per_second=103.674Gi/s items_per_second=207.348/s
// SequentialAccess/1073741824   11654268 ns     11651376 ns           51 bytes_per_second=85.8268Gi/s items_per_second=85.8268/s
//  1.42 │448:   mov          %rax,%rdx                                                                                                                                     ▒
//   1.80 │       vmovdqa      (%rax),%ymm0                                                                                                                                  ▒
//   1.72 │       vmovdqa      0x20(%rax),%ymm0                                                                                                                              ▒
//   2.25 │       add          $0x400,%rax                                                                                                                                   ▒
//   2.59 │       vmovdqa      -0x3c0(%rax),%ymm0                                                                                                                            ◆
//   2.43 │       vmovdqa      -0x3a0(%rax),%ymm0                                                                                                                            ▒
//   3.15 │       vmovdqa      -0x380(%rax),%ymm0
//   ...
//  3.08 │    │  vmovdqa      -0xc0(%rax),%ymm0                                                                                                                             ▒
//   3.06 │    │  vmovdqa      -0xa0(%rax),%ymm0                                                                                                                             ▒
//   2.97 │    │  vmovdqa      -0x80(%rax),%ymm0                                                                                                                             ▒
//   2.97 │    │  vmovdqa      -0x60(%rax),%ymm0                                                                                                                             ▒
//   3.44 │    │  vmovdqa      -0x40(%rax),%ymm0                                                                                                                             ▒
//   1.42 │    │  vmovdqa      0x3e0(%rdx),%ymm0                                                                                                                             ▒
//   1.47 │    │  vmovdqa      %ymm0,0x60(%rsp)                                                                                                                              ▒
//   1.47 │    ├──cmp          %rax,%rcx                                                                                                                                     ▒
//        │    └──jne          448 


// clang++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// SequentialAccess/1024             4.02 ns         4.02 ns    182480345 bytes_per_second=237.369Gi/s items_per_second=248.9M/s
// SequentialAccess/2048             7.78 ns         7.78 ns     89444277 bytes_per_second=245.07Gi/s items_per_second=128.487M/s
// SequentialAccess/4096             15.5 ns         15.5 ns     45121081 bytes_per_second=245.633Gi/s items_per_second=64.3913M/s
// SequentialAccess/8192             30.6 ns         30.6 ns     22824070 bytes_per_second=248.98Gi/s items_per_second=32.6343M/s
// SequentialAccess/16384            87.4 ns         87.4 ns      8152140 bytes_per_second=174.562Gi/s items_per_second=11.4401M/s
// SequentialAccess/32768             189 ns          189 ns      3669208 bytes_per_second=161.18Gi/s items_per_second=5.28155M/s  L1 1 instance
// SequentialAccess/65536             424 ns          424 ns      1678511 bytes_per_second=144.07Gi/s items_per_second=2.36044M/s
// SequentialAccess/131072            927 ns          928 ns       752530 bytes_per_second=131.599Gi/s items_per_second=1.07806M/s
// SequentialAccess/262144           1956 ns         1956 ns       359523 bytes_per_second=124.797Gi/s items_per_second=511.168k/s L1 MAX
// SequentialAccess/524288           3975 ns         3975 ns       179015 bytes_per_second=122.824Gi/s items_per_second=251.543k/s L2 1 instance
// SequentialAccess/1048576          7962 ns         7963 ns        85796 bytes_per_second=122.644Gi/s items_per_second=125.587k/s
// SequentialAccess/2097152         15972 ns        15976 ns        43962 bytes_per_second=122.257Gi/s items_per_second=62.5955k/s L3 1 instance
// SequentialAccess/4194304         31763 ns        31768 ns        22207 bytes_per_second=122.96Gi/s items_per_second=31.4778k/s  L2 MAX
// SequentialAccess/8388608         63236 ns        63247 ns        10971 bytes_per_second=123.523Gi/s items_per_second=15.811k/s  L3 MAX
// SequentialAccess/16777216       128892 ns       128898 ns         5452 bytes_per_second=121.22Gi/s items_per_second=7.75807k/s
// SequentialAccess/33554432       261138 ns       261179 ns         2681 bytes_per_second=119.65Gi/s items_per_second=3.82879k/s
// SequentialAccess/67108864       523325 ns       523376 ns         1098 bytes_per_second=119.417Gi/s items_per_second=1.91067k/s
// SequentialAccess/134217728     1072953 ns      1073041 ns          562 bytes_per_second=116.491Gi/s items_per_second=931.931/s
// SequentialAccess/268435456     2239187 ns      2239454 ns          285 bytes_per_second=111.634Gi/s items_per_second=446.538/s
// SequentialAccess/536870912     5207778 ns      5208074 ns          113 bytes_per_second=96.0048Gi/s items_per_second=192.01/s
// SequentialAccess/1073741824   11850017 ns     11851574 ns           49 bytes_per_second=84.377Gi/s items_per_second=84.377/s
// The code is the same as g++

// Note1 : L1d 256 KB (8 instances)   32  KB each
//         L2    4 MB (8 instances)   512 KB each
//         L3    8 MB (2 instances)   4   MB each

BENCHMARK_MAIN();