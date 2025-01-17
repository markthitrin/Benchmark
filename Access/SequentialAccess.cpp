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
  __m256i fill0; memset(&fill0, 0x1b, sizeof(fill0));
  __m256i fill = fill0;

  for(auto _ : state) {
    volatile __m256i* p = p0;
    while(p != end) {
      REPEAT(*p++ = fill;)
    }
    benchmark::ClobberMemory();
  }
  benchmark::DoNotOptimize(fill);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(__m256i) * array_size);
  free(memory);
}

// range(0) is memory size in byte
BENCHMARK(SequentialAccess)->RangeMultiplier(2)->Range(8192, 1<<30);

// g++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// SequentialAccess/8192              142 ns          142 ns      4933352 bytes_per_second=53.7281Gi/s items_per_second=7.04225M/s
// SequentialAccess/16384             282 ns          282 ns      2479710 bytes_per_second=54.0556Gi/s items_per_second=3.54259M/s
// SequentialAccess/32768             563 ns          563 ns      1242714 bytes_per_second=54.2293Gi/s items_per_second=1.77699M/s L1 1 instance
// SequentialAccess/65536            1187 ns         1186 ns       586079 bytes_per_second=51.4548Gi/s items_per_second=843.035k/s
// SequentialAccess/131072           2387 ns         2386 ns       293112 bytes_per_second=51.1579Gi/s items_per_second=419.085k/s
// SequentialAccess/262144           4784 ns         4783 ns       146118 bytes_per_second=51.0416Gi/s items_per_second=209.066k/s L1 MAX
// SequentialAccess/524288          10239 ns        10237 ns        68098 bytes_per_second=47.698Gi/s items_per_second=97.6855k/s  L2 1 instance
// SequentialAccess/1048576         20918 ns        20911 ns        34305 bytes_per_second=46.6999Gi/s items_per_second=47.8207k/s
// SequentialAccess/2097152         41397 ns        41393 ns        16869 bytes_per_second=47.1853Gi/s items_per_second=24.1589k/s L3 1 instance
// SequentialAccess/4194304        230183 ns       230104 ns         2750 bytes_per_second=16.976Gi/s items_per_second=4.34586k/s  L2 MAX
// SequentialAccess/8388608       1372159 ns      1371801 ns          509 bytes_per_second=5.69507Gi/s items_per_second=728.969/s  L3 MAX
// SequentialAccess/16777216      2754167 ns      2753008 ns          252 bytes_per_second=5.67561Gi/s items_per_second=363.239/s
// SequentialAccess/33554432      5693186 ns      5690855 ns          105 bytes_per_second=5.49127Gi/s items_per_second=175.721/s
// SequentialAccess/67108864     11516571 ns     11512871 ns           53 bytes_per_second=5.42871Gi/s items_per_second=86.8593/s
// SequentialAccess/134217728    24060708 ns     24053132 ns           26 bytes_per_second=5.19683Gi/s items_per_second=41.5746/s
// SequentialAccess/268435456    51426190 ns     51411308 ns           12 bytes_per_second=4.86274Gi/s items_per_second=19.451/s
// SequentialAccess/536870912   124084071 ns    124053609 ns            5 bytes_per_second=4.03052Gi/s items_per_second=8.06103/s
// SequentialAccess/1073741824  552219093 ns    552026631 ns            1 bytes_per_second=1.81151Gi/s items_per_second=1.81151/s
// The asm code move from memory to register and move regiter back to memory

// clang++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// SequentialAccess/8192              122 ns          122 ns      5713855 bytes_per_second=62.4551Gi/s items_per_second=8.18612M/s
// SequentialAccess/16384             244 ns          244 ns      2866419 bytes_per_second=62.4969Gi/s items_per_second=4.0958M/s
// SequentialAccess/32768             489 ns          489 ns      1430550 bytes_per_second=62.3838Gi/s items_per_second=2.04419M/s L1 1 instance
// SequentialAccess/65536             990 ns          990 ns       705414 bytes_per_second=61.6636Gi/s items_per_second=1.0103M/s
// SequentialAccess/131072           1983 ns         1982 ns       353119 bytes_per_second=61.5755Gi/s items_per_second=504.426k/s
// SequentialAccess/262144           3977 ns         3976 ns       175927 bytes_per_second=61.3979Gi/s items_per_second=251.486k/s L1 MAX
// SequentialAccess/524288           8175 ns         8173 ns        85546 bytes_per_second=59.7409Gi/s items_per_second=122.349k/s L2 1 instance
// SequentialAccess/1048576         16415 ns        16412 ns        42442 bytes_per_second=59.5018Gi/s items_per_second=60.9298k/s
// SequentialAccess/2097152         32599 ns        32590 ns        21448 bytes_per_second=59.9301Gi/s items_per_second=30.6842k/s L3 1 instance
// SequentialAccess/4194304        229937 ns       229851 ns         3126 bytes_per_second=16.9947Gi/s items_per_second=4.35065k/s L2 MAX
// SequentialAccess/8388608       1406084 ns      1405421 ns          474 bytes_per_second=5.55883Gi/s items_per_second=711.531/s  L3 MAX
// SequentialAccess/16777216      2781904 ns      2781236 ns          253 bytes_per_second=5.61801Gi/s items_per_second=359.552/s
// SequentialAccess/33554432      5654186 ns      5651710 ns          107 bytes_per_second=5.5293Gi/s items_per_second=176.938/s
// SequentialAccess/67108864     11458795 ns     11455273 ns           53 bytes_per_second=5.456Gi/s items_per_second=87.296/s
// SequentialAccess/134217728    23473799 ns     23464735 ns           27 bytes_per_second=5.32714Gi/s items_per_second=42.6171/s
// SequentialAccess/268435456    52659853 ns     52630004 ns           11 bytes_per_second=4.75014Gi/s items_per_second=19.0006/s
// SequentialAccess/536870912   124268362 ns    124209647 ns            5 bytes_per_second=4.02545Gi/s items_per_second=8.0509/s
// SequentialAccess/1073741824  560777918 ns    560536836 ns            1 bytes_per_second=1.784Gi/s items_per_second=1.784/s
// The asm code only move from memory to register

// Note1 : L1d 256 KB (8 instances)   32  KB each
//         L2    4 MB (8 instances)   512 KB each
//         L3    8 MB (2 instances)   2   MB each



BENCHMARK_MAIN();