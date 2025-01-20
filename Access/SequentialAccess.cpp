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
BENCHMARK(SequentialAccess)->RangeMultiplier(2)->Range(8192, 1 << 30);

// g++
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// SequentialAccess/8192              127 ns          127 ns      5499738 bytes_per_second=59.9663Gi/s items_per_second=7.8599M/s
// SequentialAccess/16384             279 ns          279 ns      2512749 bytes_per_second=54.71Gi/s items_per_second=3.58547M/s
// SequentialAccess/32768             568 ns          568 ns      1229609 bytes_per_second=53.7536Gi/s items_per_second=1.7614M/s  L1 1 instance
// SequentialAccess/65536            1147 ns         1147 ns       610638 bytes_per_second=53.2308Gi/s items_per_second=872.133k/s
// SequentialAccess/131072           2315 ns         2315 ns       301872 bytes_per_second=52.7237Gi/s items_per_second=431.913k/s
// SequentialAccess/262144           4688 ns         4687 ns       149400 bytes_per_second=52.0846Gi/s items_per_second=213.339k/s L1 MAX
// SequentialAccess/524288           9384 ns         9383 ns        74517 bytes_per_second=52.0373Gi/s items_per_second=106.572k/s L2 1 instance
// SequentialAccess/1048576         18775 ns        18773 ns        37293 bytes_per_second=52.0185Gi/s items_per_second=53.2669k/s
// SequentialAccess/2097152         37540 ns        37538 ns        18638 bytes_per_second=52.03Gi/s items_per_second=26.6394k/s   L3 1 instance
// SequentialAccess/4194304         75081 ns        75075 ns         9294 bytes_per_second=52.0315Gi/s items_per_second=13.3201k/s L2 MAX
// SequentialAccess/8388608        150309 ns       150298 ns         4653 bytes_per_second=51.9802Gi/s items_per_second=6.65347k/s L3 MAX
// SequentialAccess/16777216       301917 ns       301898 ns         2321 bytes_per_second=51.7559Gi/s items_per_second=3.31237k/s
// SequentialAccess/33554432       586911 ns       586836 ns          995 bytes_per_second=53.2517Gi/s items_per_second=1.70405k/s
// SequentialAccess/67108864      1188031 ns      1187900 ns          550 bytes_per_second=52.6139Gi/s items_per_second=841.822/s
// SequentialAccess/134217728     2449961 ns      2449466 ns          269 bytes_per_second=51.0315Gi/s items_per_second=408.252/s
// SequentialAccess/268435456     5047915 ns      5047286 ns          135 bytes_per_second=49.5316Gi/s items_per_second=198.126/s
// SequentialAccess/536870912    11524489 ns     11522994 ns           57 bytes_per_second=43.3915Gi/s items_per_second=86.783/s
// SequentialAccess/1073741824   29283412 ns     29280070 ns           18 bytes_per_second=34.1529Gi/s items_per_second=34.1529/s
// The asm code move from memory to register and move regiter back to memory

// clang++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// SequentialAccess/8192             61.5 ns         61.5 ns     11276896 bytes_per_second=124.04Gi/s items_per_second=16.2582M/s
// SequentialAccess/16384             123 ns          123 ns      5687290 bytes_per_second=124.306Gi/s items_per_second=8.14653M/s
// SequentialAccess/32768             245 ns          245 ns      2851548 bytes_per_second=124.349Gi/s items_per_second=4.07468M/s L1 1 instance
// SequentialAccess/65536             491 ns          491 ns      1424818 bytes_per_second=124.351Gi/s items_per_second=2.03737M/s 
// SequentialAccess/131072            990 ns          990 ns       679151 bytes_per_second=123.298Gi/s items_per_second=1.01005M/s
// SequentialAccess/262144           2150 ns         2150 ns       329704 bytes_per_second=113.572Gi/s items_per_second=465.19k/s  L1 MAX
// SequentialAccess/524288           4268 ns         4268 ns       174526 bytes_per_second=114.401Gi/s items_per_second=234.294k/s L2 1 instance
// SequentialAccess/1048576          8110 ns         8110 ns        81399 bytes_per_second=120.407Gi/s items_per_second=123.297k/s
// SequentialAccess/2097152         16640 ns        16639 ns        43207 bytes_per_second=117.381Gi/s items_per_second=60.0989k/s L3 1 instance
// SequentialAccess/4194304         34225 ns        34226 ns        20519 bytes_per_second=114.132Gi/s items_per_second=29.2179k/s L2 MAX
// SequentialAccess/8388608         64104 ns        64106 ns        10728 bytes_per_second=121.869Gi/s items_per_second=15.5993k/s L3 MAX
// SequentialAccess/16777216       140043 ns       140036 ns         4997 bytes_per_second=111.579Gi/s items_per_second=7.14104k/s
// SequentialAccess/33554432       281852 ns       281857 ns         2456 bytes_per_second=110.872Gi/s items_per_second=3.54789k/s
// SequentialAccess/67108864       570087 ns       570023 ns         1038 bytes_per_second=109.645Gi/s items_per_second=1.75431k/s
// SequentialAccess/134217728     1164443 ns      1164406 ns          521 bytes_per_second=107.351Gi/s items_per_second=858.807/s
// SequentialAccess/268435456     2412772 ns      2412739 ns          262 bytes_per_second=103.617Gi/s items_per_second=414.467/s
// SequentialAccess/536870912     5338771 ns      5338683 ns          105 bytes_per_second=93.6561Gi/s items_per_second=187.312/s
// SequentialAccess/1073741824   12789972 ns     12789895 ns           47 bytes_per_second=78.1867Gi/s items_per_second=78.1867/s
// The asm code only move from memory to register

// Note1 : L1d 256 KB (8 instances)   32  KB each
//         L2    4 MB (8 instances)   512 KB each
//         L3    8 MB (2 instances)   2   MB each

BENCHMARK_MAIN();