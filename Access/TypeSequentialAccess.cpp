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
// SequentialAccess<char>/16384              1947 ns         1947 ns       359379 bytes_per_second=7.83791Gi/s items_per_second=513.665k/s
// SequentialAccess<char>/32768              3900 ns         3899 ns       176241 bytes_per_second=7.82604Gi/s items_per_second=256.444k/s
// SequentialAccess<char>/65536              7833 ns         7832 ns        88622 bytes_per_second=7.79311Gi/s items_per_second=127.682k/s
// SequentialAccess<char>/131072            15647 ns        15642 ns        44625 bytes_per_second=7.80388Gi/s items_per_second=63.9294k/s
// SequentialAccess<char>/262144            31302 ns        31297 ns        22369 bytes_per_second=7.80084Gi/s items_per_second=31.9522k/s
// SequentialAccess<char>/524288            62619 ns        62607 ns        11068 bytes_per_second=7.79912Gi/s items_per_second=15.9726k/s
// SequentialAccess<char>/1048576          125564 ns       125544 ns         5551 bytes_per_second=7.77864Gi/s items_per_second=7.96533k/s

// SequentialAccess<short>/16384              982 ns          982 ns       710868 bytes_per_second=15.5406Gi/s items_per_second=1.01847M/s
// SequentialAccess<short>/32768             1960 ns         1959 ns       357136 bytes_per_second=15.5764Gi/s items_per_second=510.409k/s
// SequentialAccess<short>/65536             3896 ns         3895 ns       178865 bytes_per_second=15.6692Gi/s items_per_second=256.724k/s
// SequentialAccess<short>/131072            7807 ns         7805 ns        89421 bytes_per_second=15.6399Gi/s items_per_second=128.122k/s
// SequentialAccess<short>/262144           15658 ns        15656 ns        44594 bytes_per_second=15.5936Gi/s items_per_second=63.8713k/s
// SequentialAccess<short>/524288           31277 ns        31272 ns        22362 bytes_per_second=15.6139Gi/s items_per_second=31.9773k/s
// SequentialAccess<short>/1048576          62775 ns        62762 ns        11103 bytes_per_second=15.5597Gi/s items_per_second=15.9331k/s

// SequentialAccess<int>/16384                493 ns          493 ns      1420247 bytes_per_second=30.9449Gi/s items_per_second=2.02801M/s
// SequentialAccess<int>/32768                982 ns          982 ns       708918 bytes_per_second=31.0872Gi/s items_per_second=1.01867M/s
// SequentialAccess<int>/65536               1960 ns         1960 ns       357141 bytes_per_second=31.1461Gi/s items_per_second=510.298k/s
// SequentialAccess<int>/131072              3896 ns         3895 ns       178831 bytes_per_second=31.3415Gi/s items_per_second=256.75k/s
// SequentialAccess<int>/262144              7830 ns         7829 ns        88172 bytes_per_second=31.1851Gi/s items_per_second=127.734k/s
// SequentialAccess<int>/524288             15651 ns        15648 ns        44719 bytes_per_second=31.2042Gi/s items_per_second=63.9063k/s
// SequentialAccess<int>/1048576            31348 ns        31343 ns        22329 bytes_per_second=31.157Gi/s items_per_second=31.9047k/s

// SequentialAccess<long long>/16384          244 ns          244 ns      2864988 bytes_per_second=62.4642Gi/s items_per_second=4.09366M/s
// SequentialAccess<long long>/32768          493 ns          493 ns      1420617 bytes_per_second=61.9132Gi/s items_per_second=2.02877M/s
// SequentialAccess<long long>/65536          983 ns          983 ns       707774 bytes_per_second=62.1143Gi/s items_per_second=1.01768M/s
// SequentialAccess<long long>/131072        1961 ns         1960 ns       357068 bytes_per_second=62.2693Gi/s items_per_second=510.11k/s
// SequentialAccess<long long>/262144        3923 ns         3923 ns       178464 bytes_per_second=62.2374Gi/s items_per_second=254.925k/s
// SequentialAccess<long long>/524288        7836 ns         7835 ns        88886 bytes_per_second=62.3185Gi/s items_per_second=127.628k/s
// SequentialAccess<long long>/1048576      15682 ns        15679 ns        44561 bytes_per_second=62.2866Gi/s items_per_second=63.7815k/s

// SequentialAccess<__m128i>/16384            122 ns          122 ns      5698314 bytes_per_second=124.633Gi/s items_per_second=8.16797M/s
// SequentialAccess<__m128i>/32768            245 ns          245 ns      2857921 bytes_per_second=124.62Gi/s items_per_second=4.08355M/s
// SequentialAccess<__m128i>/65536            499 ns          499 ns      1406402 bytes_per_second=122.378Gi/s items_per_second=2.00504M/s
// SequentialAccess<__m128i>/131072           991 ns          991 ns       704859 bytes_per_second=123.188Gi/s items_per_second=1.00916M/s
// SequentialAccess<__m128i>/262144          1993 ns         1992 ns       351973 bytes_per_second=122.55Gi/s items_per_second=501.967k/s
// SequentialAccess<__m128i>/524288          3964 ns         3963 ns       176222 bytes_per_second=123.207Gi/s items_per_second=252.329k/s
// SequentialAccess<__m128i>/1048576         7957 ns         7956 ns        87655 bytes_per_second=122.742Gi/s items_per_second=125.687k/s

// SequentialAccess<__m256i>/16384            104 ns          104 ns      6742519 bytes_per_second=147.397Gi/s items_per_second=9.65982M/s
// SequentialAccess<__m256i>/32768            207 ns          207 ns      3377976 bytes_per_second=147.34Gi/s items_per_second=4.82804M/s
// SequentialAccess<__m256i>/65536            433 ns          433 ns      1617131 bytes_per_second=141.024Gi/s items_per_second=2.31054M/s
// SequentialAccess<__m256i>/131072           907 ns          907 ns       767056 bytes_per_second=134.559Gi/s items_per_second=1.10231M/s
// SequentialAccess<__m256i>/262144          1907 ns         1908 ns       367185 bytes_per_second=127.951Gi/s items_per_second=524.089k/s
// SequentialAccess<__m256i>/524288          3896 ns         3898 ns       179567 bytes_per_second=125.267Gi/s items_per_second=256.546k/s
// SequentialAccess<__m256i>/1048576         7817 ns         7821 ns        89092 bytes_per_second=124.866Gi/s items_per_second=127.862k/s

// SequentialAccess<__m512i>/16384            103 ns          104 ns      6735669 bytes_per_second=147.381Gi/s items_per_second=9.65877M/s
// SequentialAccess<__m512i>/32768            207 ns          207 ns      3380782 bytes_per_second=147.467Gi/s items_per_second=4.83219M/s
// SequentialAccess<__m512i>/65536            431 ns          431 ns      1618407 bytes_per_second=141.665Gi/s items_per_second=2.32104M/s
// SequentialAccess<__m512i>/131072           896 ns          897 ns       778623 bytes_per_second=136.154Gi/s items_per_second=1.11537M/s
// SequentialAccess<__m512i>/262144          1904 ns         1904 ns       371251 bytes_per_second=128.193Gi/s items_per_second=525.08k/s
// SequentialAccess<__m512i>/524288          3865 ns         3867 ns       181171 bytes_per_second=126.259Gi/s items_per_second=258.579k/s
// SequentialAccess<__m512i>/1048576         7761 ns         7764 ns        89974 bytes_per_second=125.778Gi/s items_per_second=128.797k/s
// The code is the same with g++, But for the 512bit it ignore moving to the stack, just move to the register, yeilding the same result as 256bit
BENCHMARK_MAIN();