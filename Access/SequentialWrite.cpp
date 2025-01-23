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

static void SequentialWrite(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(__m256i);
  if(posix_memalign(&memory, 64, array_size * sizeof(__m256i)) != 0)
    abort();
  volatile __m256i* const p0 = static_cast<__m256i*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(__m256i);
  __m256i fill0; memset(&fill0, 0x1b, sizeof(fill0));
  __m256i fill = fill0;

  for(auto _ : state) {
    volatile __m256i* p = p0;
    while(p != end) {
      REPEAT(*p++ = fill;)
    }
  }
  benchmark::DoNotOptimize(fill);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size);
  free(memory);
}

// range(0) is memory size in byte
BENCHMARK(SequentialWrite)->RangeMultiplier(2)->Range(1024, 1 << 30);

// g++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// SequentialWrite/1024             7.65 ns         7.64 ns     91712035 bytes_per_second=124.773Gi/s items_per_second=130.834M/s
// SequentialWrite/2048             15.3 ns         15.2 ns     45899888 bytes_per_second=125.092Gi/s items_per_second=65.5842M/s
// SequentialWrite/4096             30.5 ns         30.5 ns     22974783 bytes_per_second=125.185Gi/s items_per_second=32.8166M/s
// SequentialWrite/8192              136 ns          136 ns      5140598 bytes_per_second=56.0259Gi/s items_per_second=7.34342M/s
// SequentialWrite/16384             272 ns          272 ns      2573192 bytes_per_second=56.0758Gi/s items_per_second=3.67499M/s
// SequentialWrite/32768             545 ns          545 ns      1279066 bytes_per_second=55.9853Gi/s items_per_second=1.83453M/s
// SequentialWrite/65536            1117 ns         1117 ns       626171 bytes_per_second=54.6553Gi/s items_per_second=895.472k/s
// SequentialWrite/131072           2231 ns         2231 ns       313748 bytes_per_second=54.7113Gi/s items_per_second=448.195k/s
// SequentialWrite/262144           4448 ns         4448 ns       157221 bytes_per_second=54.8867Gi/s items_per_second=224.816k/s
// SequentialWrite/524288           9172 ns         9172 ns        75966 bytes_per_second=53.2375Gi/s items_per_second=109.03k/s
// SequentialWrite/1048576         18980 ns        18980 ns        36939 bytes_per_second=51.4529Gi/s items_per_second=52.6877k/s
// SequentialWrite/2097152         39670 ns        39666 ns        17664 bytes_per_second=49.2398Gi/s items_per_second=25.2108k/s
// SequentialWrite/4194304        227061 ns       227036 ns         2811 bytes_per_second=17.2054Gi/s items_per_second=4.40459k/s
// SequentialWrite/8388608       1361240 ns      1361112 ns          510 bytes_per_second=5.73979Gi/s items_per_second=734.693/s
// SequentialWrite/16777216      2730071 ns      2729982 ns          255 bytes_per_second=5.72348Gi/s items_per_second=366.303/s
// SequentialWrite/33554432      5772578 ns      5772447 ns           99 bytes_per_second=5.41365Gi/s items_per_second=173.237/s
// SequentialWrite/67108864     11334897 ns     11332051 ns           53 bytes_per_second=5.51533Gi/s items_per_second=88.2453/s
// SequentialWrite/134217728    23500073 ns     23495680 ns           26 bytes_per_second=5.32013Gi/s items_per_second=42.561/s
// SequentialWrite/268435456    51642456 ns     51626001 ns           11 bytes_per_second=4.84252Gi/s items_per_second=19.3701/s
// SequentialWrite/536870912   122606043 ns    122601258 ns            5 bytes_per_second=4.07826Gi/s items_per_second=8.15652/s
// SequentialWrite/1073741824  648162884 ns    643595423 ns            1 bytes_per_second=1.55377Gi/s items_per_second=1.55377/s
// Put in register and then pass to memory

// clang++
// -------------------------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations UserCounters...
// -------------------------------------------------------------------------------------
// SequentialWrite/1024             7.70 ns         7.70 ns     90890749 bytes_per_second=123.85Gi/s items_per_second=129.866M/s
// SequentialWrite/2048             15.3 ns         15.3 ns     45482913 bytes_per_second=124.477Gi/s items_per_second=65.2618M/s
// SequentialWrite/4096             30.6 ns         30.6 ns     22814797 bytes_per_second=124.522Gi/s items_per_second=32.6426M/s
// SequentialWrite/8192              122 ns          122 ns      5695297 bytes_per_second=62.5916Gi/s items_per_second=8.204M/s
// SequentialWrite/16384             243 ns          243 ns      2882671 bytes_per_second=62.8649Gi/s items_per_second=4.11991M/s
// SequentialWrite/32768             487 ns          487 ns      1441209 bytes_per_second=62.7023Gi/s items_per_second=2.05463M/s
// SequentialWrite/65536             984 ns          984 ns       709531 bytes_per_second=62.0355Gi/s items_per_second=1.01639M/s
// SequentialWrite/131072           1969 ns         1969 ns       354793 bytes_per_second=61.9876Gi/s items_per_second=507.803k/s
// SequentialWrite/262144           3942 ns         3942 ns       177457 bytes_per_second=61.9341Gi/s items_per_second=253.682k/s
// SequentialWrite/524288           8451 ns         8451 ns        82133 bytes_per_second=57.7805Gi/s items_per_second=118.334k/s
// SequentialWrite/1048576         16294 ns        16291 ns        42914 bytes_per_second=59.9436Gi/s items_per_second=61.3823k/s
// SequentialWrite/2097152         32636 ns        32635 ns        21448 bytes_per_second=59.847Gi/s items_per_second=30.6417k/s
// SequentialWrite/4194304        228556 ns       228506 ns         3050 bytes_per_second=17.0947Gi/s items_per_second=4.37625k/s
// SequentialWrite/8388608       1354952 ns      1354758 ns          506 bytes_per_second=5.76671Gi/s items_per_second=738.139/s
// SequentialWrite/16777216      2726386 ns      2725758 ns          255 bytes_per_second=5.73235Gi/s items_per_second=366.87/s
// SequentialWrite/33554432      5595228 ns      5594419 ns          106 bytes_per_second=5.58592Gi/s items_per_second=178.75/s
// SequentialWrite/67108864     11258817 ns     11255239 ns           53 bytes_per_second=5.55297Gi/s items_per_second=88.8475/s
// SequentialWrite/134217728    23437394 ns     23430091 ns           26 bytes_per_second=5.33502Gi/s items_per_second=42.6802/s
// SequentialWrite/268435456    53519596 ns     53504842 ns           11 bytes_per_second=4.67247Gi/s items_per_second=18.6899/s
// SequentialWrite/536870912   132169078 ns    132165424 ns            4 bytes_per_second=3.78314Gi/s items_per_second=7.56628/s
// SequentialWrite/1073741824  555771172 ns    555628499 ns            1 bytes_per_second=1.79976Gi/s items_per_second=1.79976/s
// Copy once, always move.

// Note1 : L1d 256 KB (8 instances)   32  KB each
//         L2    4 MB (8 instances)   512 KB each
//         L3    8 MB (2 instances)   2   MB each

BENCHMARK_MAIN();