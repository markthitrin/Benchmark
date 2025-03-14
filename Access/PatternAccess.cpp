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

static void BackwardAccess(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  void* const rend = static_cast<char*>(memory) - 1 * sizeof(__m256i);
  volatile __m256i* const p0 = static_cast<__m256i*>(memory) + array_size - 1;
  __m256i sink0; memset(&sink0, 0x1b, sizeof(sink0));
  __m256i sink = sink0;

  for(auto _ : state) {
    volatile __m256i* p = p0;
    while(p != rend) {
      REPEAT(sink = *p--;)
    }
  }
  benchmark::DoNotOptimize(sink);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size);
  free(memory);
}

static void SequentialIndexAccess(benchmark::State& state) {
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

static void Skip1Access(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  volatile __m256i* const p0 = static_cast<__m256i*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(__m256i);
  __m256i sink0; memset(&sink0, 0x1b, sizeof(sink0));
  __m256i sink = sink0;

  if(array_size == 32) {
    for(auto _ : state) {
      volatile __m256i* p = p0;
      while(p != end) {
        REPEAT16(sink = *p; p += 2;);
      }
    }
    benchmark::DoNotOptimize(sink);
  }
  else if(array_size >= 64) {
    for(auto _ : state) {
      volatile __m256i* p = p0;
      while(p != end) {
        REPEAT(sink = *p; p += 2;);
      }
    }
    benchmark::DoNotOptimize(sink);
  }

  state.SetItemsProcessed(state.iterations() / 2);
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size / 2);
  free(memory);
}

BENCHMARK(BackwardAccess)->RangeMultiplier(2)->Range(1024, 1 << 30);
BENCHMARK(SequentialIndexAccess)->RangeMultiplier(2)->Range(1024, 1 << 30);
BENCHMARK(Skip1Access)->RangeMultiplier(2)->Range(1024, 1 << 30);



// g++
// -------------------------------------------------------------------------------------------
// Benchmark                                 Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------------
// BackwardAccess/1024                    7.74 ns         7.74 ns     90570135 bytes_per_second=123.267Gi/s items_per_second=129.254M/s
// BackwardAccess/2048                    15.5 ns         15.5 ns     45174840 bytes_per_second=123.366Gi/s items_per_second=64.6791M/s
// BackwardAccess/4096                    30.9 ns         30.9 ns     22641779 bytes_per_second=123.427Gi/s items_per_second=32.3556M/s
// BackwardAccess/8192                    61.5 ns         61.5 ns     11304957 bytes_per_second=124.039Gi/s items_per_second=16.258M/s
// BackwardAccess/16384                    138 ns          138 ns      5078930 bytes_per_second=110.845Gi/s items_per_second=7.26432M/s
// BackwardAccess/32768                    285 ns          285 ns      2461868 bytes_per_second=107.026Gi/s items_per_second=3.50701M/s
// BackwardAccess/65536                    586 ns          586 ns      1185467 bytes_per_second=104.209Gi/s items_per_second=1.70736M/s
// BackwardAccess/131072                  1210 ns         1210 ns       575241 bytes_per_second=100.868Gi/s items_per_second=826.312k/s
// BackwardAccess/262144                  2470 ns         2470 ns       282632 bytes_per_second=98.8587Gi/s items_per_second=404.925k/s
// BackwardAccess/524288                  4959 ns         4958 ns       141253 bytes_per_second=98.4839Gi/s items_per_second=201.695k/s
// BackwardAccess/1048576                 9913 ns         9913 ns        70339 bytes_per_second=98.514Gi/s items_per_second=100.878k/s
// BackwardAccess/2097152                19832 ns        19830 ns        35154 bytes_per_second=98.4916Gi/s items_per_second=50.4277k/s
// BackwardAccess/4194304                39698 ns        39694 ns        17565 bytes_per_second=98.4088Gi/s items_per_second=25.1927k/s
// BackwardAccess/8388608                79540 ns        79533 ns         8782 bytes_per_second=98.23Gi/s items_per_second=12.5734k/s
// BackwardAccess/16777216              160447 ns       160441 ns         4295 bytes_per_second=97.3879Gi/s items_per_second=6.23282k/s
// BackwardAccess/33554432              327663 ns       327623 ns         2140 bytes_per_second=95.384Gi/s items_per_second=3.05229k/s
// BackwardAccess/67108864              662153 ns       662124 ns          916 bytes_per_second=94.3932Gi/s items_per_second=1.51029k/s
// BackwardAccess/134217728            1348257 ns      1348206 ns          458 bytes_per_second=92.7158Gi/s items_per_second=741.726/s
// BackwardAccess/268435456            2770276 ns      2770058 ns          229 bytes_per_second=90.2508Gi/s items_per_second=361.003/s
// BackwardAccess/536870912            6160880 ns      6160315 ns           93 bytes_per_second=81.1647Gi/s items_per_second=162.329/s
// BackwardAccess/1073741824          14708659 ns     14707555 ns           42 bytes_per_second=67.9923Gi/s items_per_second=67.9923/s
// SequentialIndexAccess/1024             9.43 ns         9.43 ns     74293631 bytes_per_second=101.112Gi/s items_per_second=106.024M/s
// SequentialIndexAccess/2048             18.4 ns         18.4 ns     38056887 bytes_per_second=103.924Gi/s items_per_second=54.4861M/s
// SequentialIndexAccess/4096             36.7 ns         36.7 ns     19010853 bytes_per_second=103.817Gi/s items_per_second=27.215M/s
// SequentialIndexAccess/8192             73.1 ns         73.0 ns      9477872 bytes_per_second=104.441Gi/s items_per_second=13.6893M/s
// SequentialIndexAccess/16384             175 ns          175 ns      4016538 bytes_per_second=87.1667Gi/s items_per_second=5.71256M/s
// SequentialIndexAccess/32768             358 ns          358 ns      1957258 bytes_per_second=85.3394Gi/s items_per_second=2.7964M/s
// SequentialIndexAccess/65536             722 ns          722 ns       966669 bytes_per_second=84.587Gi/s items_per_second=1.38587M/s
// SequentialIndexAccess/131072           1458 ns         1458 ns       480136 bytes_per_second=83.7474Gi/s items_per_second=686.058k/s
// SequentialIndexAccess/262144           2955 ns         2955 ns       238186 bytes_per_second=82.6225Gi/s items_per_second=338.422k/s
// SequentialIndexAccess/524288           5933 ns         5932 ns       117462 bytes_per_second=82.3078Gi/s items_per_second=168.566k/s
// SequentialIndexAccess/1048576         11800 ns        11800 ns        59004 bytes_per_second=82.7616Gi/s items_per_second=84.7479k/s
// SequentialIndexAccess/2097152         23833 ns        23830 ns        29713 bytes_per_second=81.961Gi/s items_per_second=41.964k/s
// SequentialIndexAccess/4194304         49125 ns        49121 ns        14537 bytes_per_second=79.523Gi/s items_per_second=20.3579k/s
// SequentialIndexAccess/8388608         99348 ns        99336 ns         7175 bytes_per_second=78.6473Gi/s items_per_second=10.0669k/s
// SequentialIndexAccess/16777216       199283 ns       199258 ns         3469 bytes_per_second=78.4158Gi/s items_per_second=5.01861k/s
// SequentialIndexAccess/33554432       423351 ns       423284 ns         1656 bytes_per_second=73.8276Gi/s items_per_second=2.36248k/s
// SequentialIndexAccess/67108864       890725 ns       890641 ns          710 bytes_per_second=70.1742Gi/s items_per_second=1.12279k/s
// SequentialIndexAccess/134217728     1792028 ns      1791514 ns          357 bytes_per_second=69.7734Gi/s items_per_second=558.187/s
// SequentialIndexAccess/268435456     3702247 ns      3701431 ns          178 bytes_per_second=67.5414Gi/s items_per_second=270.166/s
// SequentialIndexAccess/536870912     8177869 ns      8176127 ns           75 bytes_per_second=61.1536Gi/s items_per_second=122.307/s
// SequentialIndexAccess/1073741824   20945482 ns     20942359 ns           25 bytes_per_second=47.7501Gi/s items_per_second=47.7501/s
// Skip1Access/1024                       2.02 ns         2.02 ns    354096068 bytes_per_second=235.747Gi/s items_per_second=247.198M/s
// Skip1Access/2048                       4.07 ns         4.07 ns    172470478 bytes_per_second=234.506Gi/s items_per_second=122.948M/s
// Skip1Access/4096                       8.16 ns         8.16 ns     81923260 bytes_per_second=233.72Gi/s items_per_second=61.2683M/s
// Skip1Access/8192                       16.4 ns         16.4 ns     42738842 bytes_per_second=232.306Gi/s items_per_second=30.4488M/s
// Skip1Access/16384                      91.4 ns         91.4 ns      7624063 bytes_per_second=83.4573Gi/s items_per_second=5.46946M/s
// Skip1Access/32768                       192 ns          192 ns      3646261 bytes_per_second=79.5367Gi/s items_per_second=2.60626M/s
// Skip1Access/65536                       457 ns          457 ns      1531288 bytes_per_second=66.7655Gi/s items_per_second=1.09389M/s
// Skip1Access/131072                      914 ns          914 ns       764605 bytes_per_second=66.7632Gi/s items_per_second=546.924k/s
// Skip1Access/262144                     1925 ns         1925 ns       363525 bytes_per_second=63.4224Gi/s items_per_second=259.777k/s
// Skip1Access/524288                     3851 ns         3851 ns       181508 bytes_per_second=63.3985Gi/s items_per_second=129.84k/s
// Skip1Access/1048576                    7784 ns         7783 ns        89763 bytes_per_second=62.7357Gi/s items_per_second=64.2407k/s
// Skip1Access/2097152                   15607 ns        15605 ns        44019 bytes_per_second=62.5798Gi/s items_per_second=32.0401k/s
// Skip1Access/4194304                   31213 ns        31210 ns        22433 bytes_per_second=62.5798Gi/s items_per_second=16.0197k/s
// Skip1Access/8388608                   62512 ns        62506 ns        11172 bytes_per_second=62.4936Gi/s items_per_second=7.99918k/s
// Skip1Access/16777216                 125754 ns       125741 ns         5580 bytes_per_second=62.1316Gi/s items_per_second=3.97642k/s
// Skip1Access/33554432                 254454 ns       254443 ns         2713 bytes_per_second=61.4086Gi/s items_per_second=1.96435k/s
// Skip1Access/67108864                 514690 ns       514632 ns         1134 bytes_per_second=60.723Gi/s items_per_second=971.568/s
// Skip1Access/134217728               1049796 ns      1049744 ns          567 bytes_per_second=59.5383Gi/s items_per_second=475.467/s
// Skip1Access/268435456               2178619 ns      2178401 ns          287 bytes_per_second=57.3815Gi/s items_per_second=228.726/s
// Skip1Access/536870912               4836280 ns      4835742 ns          112 bytes_per_second=51.6984Gi/s items_per_second=103.397/s
// Skip1Access/1073741824             11594258 ns     11593361 ns           51 bytes_per_second=43.1281Gi/s items_per_second=42.2825/s

BENCHMARK_MAIN();