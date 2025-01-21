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

template<class Word>
static void SequentialAccess(benchmark::State& state) {
  void* memory;
  const int array_size = state.range(0) / sizeof(Word);
  if(posix_memalign(&memory, 64, array_size * sizeof(Word)) != 0)
    abort();
  volatile Word* const p0 = static_cast<Word*>(memory);
  void* const end = static_cast<char*>(memory) + array_size * sizeof(Word);
  Word sink0; memset(&sink0, 0x1b, sizeof(sink0));
  Word sink = sink0;

  for(auto _ : state) {
    volatile Word* p = p0;
    while(p != end) {
      REPEAT(sink = *p++;)
    }
  }
  benchmark::DoNotOptimize(sink);

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(Word) * array_size);
  free(memory);
}

// range(0) is memory size in byte
BENCHMARK(SequentialAccess<char>)->RangeMultiplier(2)->Range(16384, 1048576);
BENCHMARK(SequentialAccess<short>)->RangeMultiplier(2)->Range(16384, 1048576);
BENCHMARK(SequentialAccess<int>)->RangeMultiplier(2)->Range(16384, 1048576);
BENCHMARK(SequentialAccess<long long>)->RangeMultiplier(2)->Range(16384, 1048576);
BENCHMARK(SequentialAccess<__m128i>)->RangeMultiplier(2)->Range(16384, 1048576);
BENCHMARK(SequentialAccess<__m256i>)->RangeMultiplier(2)->Range(16384, 1048576);
BENCHMARK(SequentialAccess<__m512i>)->RangeMultiplier(2)->Range(16384, 1048576);
// g++
// ----------------------------------------------------------------------------------------------
// Benchmark                                    Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------------------
// SequentialAccess<char>/16384              1954 ns         1953 ns       358579 bytes_per_second=7.8146Gi/s items_per_second=512.138k/s
// SequentialAccess<char>/32768              3893 ns         3891 ns       180112 bytes_per_second=7.84365Gi/s items_per_second=257.021k/s
// SequentialAccess<char>/65536              7777 ns         7772 ns        89523 bytes_per_second=7.85306Gi/s items_per_second=128.665k/s
// SequentialAccess<char>/131072            15543 ns        15533 ns        45005 bytes_per_second=7.85876Gi/s items_per_second=64.379k/s
// SequentialAccess<char>/262144            31203 ns        31185 ns        22443 bytes_per_second=7.82884Gi/s items_per_second=32.0669k/s
// SequentialAccess<char>/524288            62365 ns        62320 ns        11154 bytes_per_second=7.83507Gi/s items_per_second=16.0462k/s
// SequentialAccess<char>/1048576          124842 ns       124762 ns         5602 bytes_per_second=7.8274Gi/s items_per_second=8.01525k/s

// SequentialAccess<short>/16384              975 ns          974 ns       714958 bytes_per_second=15.6592Gi/s items_per_second=1.02624M/s
// SequentialAccess<short>/32768             1946 ns         1945 ns       359801 bytes_per_second=15.6912Gi/s items_per_second=514.17k/s
// SequentialAccess<short>/65536             3897 ns         3894 ns       180027 bytes_per_second=15.6746Gi/s items_per_second=256.813k/s
// SequentialAccess<short>/131072            7788 ns         7784 ns        88936 bytes_per_second=15.6828Gi/s items_per_second=128.474k/s
// SequentialAccess<short>/262144           15707 ns        15697 ns        44623 bytes_per_second=15.5534Gi/s items_per_second=63.7067k/s
// SequentialAccess<short>/524288           31408 ns        31386 ns        22274 bytes_per_second=15.5575Gi/s items_per_second=31.8618k/s
// SequentialAccess<short>/1048576          62862 ns        62826 ns        11063 bytes_per_second=15.5438Gi/s items_per_second=15.9169k/s

// SequentialAccess<int>/16384                490 ns          489 ns      1430571 bytes_per_second=31.1775Gi/s items_per_second=2.04325M/s
// SequentialAccess<int>/32768                976 ns          975 ns       715675 bytes_per_second=31.2926Gi/s items_per_second=1.0254M/s
// SequentialAccess<int>/65536               1946 ns         1945 ns       359687 bytes_per_second=31.3789Gi/s items_per_second=514.111k/s
// SequentialAccess<int>/131072              3889 ns         3887 ns       180082 bytes_per_second=31.4081Gi/s items_per_second=257.295k/s
// SequentialAccess<int>/262144              7800 ns         7795 ns        88871 bytes_per_second=31.3209Gi/s items_per_second=128.29k/s
// SequentialAccess<int>/524288             15598 ns        15588 ns        44910 bytes_per_second=31.3245Gi/s items_per_second=64.1525k/s
// SequentialAccess<int>/1048576            31211 ns        31191 ns        22432 bytes_per_second=31.3092Gi/s items_per_second=32.0606k/s

// SequentialAccess<long long>/16384          243 ns          243 ns      2879510 bytes_per_second=62.7891Gi/s items_per_second=4.11495M/s
// SequentialAccess<long long>/32768          491 ns          491 ns      1425231 bytes_per_second=62.1401Gi/s items_per_second=2.03621M/s
// SequentialAccess<long long>/65536          978 ns          977 ns       711310 bytes_per_second=62.4708Gi/s items_per_second=1.02352M/s
// SequentialAccess<long long>/131072        1949 ns         1947 ns       359020 bytes_per_second=62.6864Gi/s items_per_second=513.527k/s
// SequentialAccess<long long>/262144        3911 ns         3908 ns       179498 bytes_per_second=62.4662Gi/s items_per_second=255.862k/s
// SequentialAccess<long long>/524288        7804 ns         7798 ns        88853 bytes_per_second=62.6136Gi/s items_per_second=128.233k/s
// SequentialAccess<long long>/1048576      15626 ns        15617 ns        44698 bytes_per_second=62.5303Gi/s items_per_second=64.031k/s

// SequentialAccess<__m128i>/16384            127 ns          127 ns      5490383 bytes_per_second=119.787Gi/s items_per_second=7.85039M/s
// SequentialAccess<__m128i>/32768            250 ns          250 ns      2805036 bytes_per_second=122.284Gi/s items_per_second=4.00699M/s
// SequentialAccess<__m128i>/65536            495 ns          495 ns      1415157 bytes_per_second=123.404Gi/s items_per_second=2.02185M/s
// SequentialAccess<__m128i>/131072           989 ns          989 ns       708877 bytes_per_second=123.461Gi/s items_per_second=1.01139M/s
// SequentialAccess<__m128i>/262144          1979 ns         1978 ns       353870 bytes_per_second=123.444Gi/s items_per_second=505.626k/s
// SequentialAccess<__m128i>/524288          3935 ns         3933 ns       178080 bytes_per_second=124.166Gi/s items_per_second=254.291k/s
// SequentialAccess<__m128i>/1048576         7863 ns         7858 ns        88140 bytes_per_second=124.278Gi/s items_per_second=127.261k/s

// SequentialAccess<__m256i>/16384            104 ns          103 ns      6750654 bytes_per_second=147.441Gi/s items_per_second=9.66267M/s
// SequentialAccess<__m256i>/32768            207 ns          207 ns      3381240 bytes_per_second=147.342Gi/s items_per_second=4.8281M/s
// SequentialAccess<__m256i>/65536            433 ns          433 ns      1618361 bytes_per_second=141.12Gi/s items_per_second=2.31211M/s
// SequentialAccess<__m256i>/131072           906 ns          905 ns       767957 bytes_per_second=134.871Gi/s items_per_second=1.10486M/s
// SequentialAccess<__m256i>/262144          1909 ns         1907 ns       366762 bytes_per_second=128.005Gi/s items_per_second=524.307k/s
// SequentialAccess<__m256i>/524288          3865 ns         3863 ns       181283 bytes_per_second=126.408Gi/s items_per_second=258.883k/s
// SequentialAccess<__m256i>/1048576         7759 ns         7754 ns        90091 bytes_per_second=125.942Gi/s items_per_second=128.964k/s

// SequentialAccess<__m512i>/16384            156 ns          156 ns      4512106 bytes_per_second=97.9745Gi/s items_per_second=6.42086M/s
// SequentialAccess<__m512i>/32768            319 ns          319 ns      2194254 bytes_per_second=95.7367Gi/s items_per_second=3.1371M/s
// SequentialAccess<__m512i>/65536            648 ns          648 ns      1079335 bytes_per_second=94.2085Gi/s items_per_second=1.54351M/s
// SequentialAccess<__m512i>/131072          1310 ns         1310 ns       534442 bytes_per_second=93.2154Gi/s items_per_second=763.621k/s
// SequentialAccess<__m512i>/262144          2645 ns         2645 ns       265334 bytes_per_second=92.307Gi/s items_per_second=378.089k/s
// SequentialAccess<__m512i>/524288          5334 ns         5341 ns       130891 bytes_per_second=91.4225Gi/s items_per_second=187.233k/s
// SequentialAccess<__m512i>/1048576        10737 ns        10748 ns        65100 bytes_per_second=90.8582Gi/s items_per_second=93.0388k/s
// The code initially use movaps which require two of these to move 256 bit which make it two time slower. as this effect
// still occur with 512bit even enable avx since the computer doesn't support avx512


// clang++
// ----------------------------------------------------------------------------------------------
// Benchmark                                    Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------------------
// SequentialAccess<char>/16384              2044 ns         2044 ns       340692 bytes_per_second=7.46695Gi/s items_per_second=489.354k/s
// SequentialAccess<char>/32768              4058 ns         4058 ns       177280 bytes_per_second=7.52109Gi/s items_per_second=246.451k/s
// SequentialAccess<char>/65536              8341 ns         8341 ns        85958 bytes_per_second=7.31772Gi/s items_per_second=119.894k/s
// SequentialAccess<char>/131072            16696 ns        16697 ns        42291 bytes_per_second=7.31091Gi/s items_per_second=59.891k/s
// SequentialAccess<char>/262144            33165 ns        33163 ns        21076 bytes_per_second=7.36194Gi/s items_per_second=30.1545k/s
// SequentialAccess<char>/524288            64066 ns        64067 ns        10558 bytes_per_second=7.62142Gi/s items_per_second=15.6087k/s
// SequentialAccess<char>/1048576          129371 ns       129364 ns         5485 bytes_per_second=7.54895Gi/s items_per_second=7.73012k/s

// SequentialAccess<short>/16384             1007 ns         1007 ns       695446 bytes_per_second=15.1485Gi/s items_per_second=992.77k/s
// SequentialAccess<short>/32768             2032 ns         2032 ns       355843 bytes_per_second=15.0215Gi/s items_per_second=492.225k/s
// SequentialAccess<short>/65536             4069 ns         4069 ns       178333 bytes_per_second=14.9985Gi/s items_per_second=245.736k/s
// SequentialAccess<short>/131072            8036 ns         8036 ns        82235 bytes_per_second=15.1912Gi/s items_per_second=124.446k/s
// SequentialAccess<short>/262144           16428 ns        16428 ns        44201 bytes_per_second=14.8614Gi/s items_per_second=60.8722k/s
// SequentialAccess<short>/524288           33035 ns        33036 ns        22159 bytes_per_second=14.7804Gi/s items_per_second=30.2703k/s
// SequentialAccess<short>/1048576          64729 ns        64727 ns        10410 bytes_per_second=15.0874Gi/s items_per_second=15.4495k/s

// SequentialAccess<int>/16384                506 ns          506 ns      1352118 bytes_per_second=30.1737Gi/s items_per_second=1.97747M/s
// SequentialAccess<int>/32768               1019 ns         1019 ns       697582 bytes_per_second=29.9447Gi/s items_per_second=981.227k/s
// SequentialAccess<int>/65536               2065 ns         2065 ns       352043 bytes_per_second=29.5568Gi/s items_per_second=484.259k/s
// SequentialAccess<int>/131072              4105 ns         4105 ns       178060 bytes_per_second=29.7365Gi/s items_per_second=243.601k/s
// SequentialAccess<int>/262144              8090 ns         8090 ns        84275 bytes_per_second=30.1797Gi/s items_per_second=123.616k/s
// SequentialAccess<int>/524288             16324 ns        16324 ns        43911 bytes_per_second=29.9122Gi/s items_per_second=61.2603k/s
// SequentialAccess<int>/1048576            33263 ns        33261 ns        22153 bytes_per_second=29.3604Gi/s items_per_second=30.065k/s

// SequentialAccess<long long>/16384          254 ns          254 ns      2810328 bytes_per_second=60.0207Gi/s items_per_second=3.93352M/s
// SequentialAccess<long long>/32768          509 ns          509 ns      1255762 bytes_per_second=59.9454Gi/s items_per_second=1.96429M/s
// SequentialAccess<long long>/65536         1007 ns         1007 ns       694410 bytes_per_second=60.633Gi/s items_per_second=993.411k/s
// SequentialAccess<long long>/131072        2014 ns         2014 ns       353198 bytes_per_second=60.6202Gi/s items_per_second=496.601k/s
// SequentialAccess<long long>/262144        4080 ns         4080 ns       177764 bytes_per_second=59.8369Gi/s items_per_second=245.092k/s
// SequentialAccess<long long>/524288        7994 ns         7993 ns        82433 bytes_per_second=61.0922Gi/s items_per_second=125.117k/s
// SequentialAccess<long long>/1048576      16530 ns        16529 ns        43551 bytes_per_second=59.0803Gi/s items_per_second=60.4982k/s

// SequentialAccess<__m128i>/16384            132 ns          132 ns      5268456 bytes_per_second=115.379Gi/s items_per_second=7.56146M/s
// SequentialAccess<__m128i>/32768            264 ns          264 ns      2757596 bytes_per_second=115.614Gi/s items_per_second=3.78845M/s
// SequentialAccess<__m128i>/65536            506 ns          506 ns      1313177 bytes_per_second=120.666Gi/s items_per_second=1.97699M/s
// SequentialAccess<__m128i>/131072          1016 ns         1016 ns       691668 bytes_per_second=120.116Gi/s items_per_second=983.99k/s
// SequentialAccess<__m128i>/262144          2046 ns         2046 ns       347834 bytes_per_second=119.32Gi/s items_per_second=488.736k/s
// SequentialAccess<__m128i>/524288          4300 ns         4300 ns       175906 bytes_per_second=113.556Gi/s items_per_second=232.563k/s
// SequentialAccess<__m128i>/1048576         8073 ns         8073 ns        83681 bytes_per_second=120.97Gi/s items_per_second=123.874k/s

// SequentialAccess<__m256i>/16384            105 ns          105 ns      6610419 bytes_per_second=144.89Gi/s items_per_second=9.49552M/s
// SequentialAccess<__m256i>/32768            211 ns          211 ns      3354990 bytes_per_second=144.9Gi/s items_per_second=4.74807M/s
// SequentialAccess<__m256i>/65536            442 ns          442 ns      1610135 bytes_per_second=138.165Gi/s items_per_second=2.26369M/s
// SequentialAccess<__m256i>/131072           911 ns          911 ns       755507 bytes_per_second=134.056Gi/s items_per_second=1.09819M/s
// SequentialAccess<__m256i>/262144          1904 ns         1904 ns       368914 bytes_per_second=128.203Gi/s items_per_second=525.121k/s
// SequentialAccess<__m256i>/524288          3932 ns         3933 ns       180218 bytes_per_second=124.163Gi/s items_per_second=254.286k/s
// SequentialAccess<__m256i>/1048576         7819 ns         7819 ns        87791 bytes_per_second=124.904Gi/s items_per_second=127.902k/s

// SequentialAccess<__m512i>/16384            163 ns          163 ns      4411356 bytes_per_second=93.3578Gi/s items_per_second=6.1183M/s
// SequentialAccess<__m512i>/32768            338 ns          338 ns      2127544 bytes_per_second=90.2828Gi/s items_per_second=2.95839M/s
// SequentialAccess<__m512i>/65536            682 ns          682 ns       995598 bytes_per_second=89.5505Gi/s items_per_second=1.46719M/s
// SequentialAccess<__m512i>/131072          1383 ns         1383 ns       509318 bytes_per_second=88.2561Gi/s items_per_second=722.994k/s
// SequentialAccess<__m512i>/262144          2799 ns         2799 ns       254636 bytes_per_second=87.2191Gi/s items_per_second=357.249k/s
// SequentialAccess<__m512i>/524288          5503 ns         5503 ns       124224 bytes_per_second=88.7351Gi/s items_per_second=181.729k/s
// SequentialAccess<__m512i>/1048576        11033 ns        11032 ns        63259 bytes_per_second=88.5218Gi/s items_per_second=90.6463k/s
// The code is the same with g++
BENCHMARK_MAIN();