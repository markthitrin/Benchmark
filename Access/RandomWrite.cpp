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
BENCHMARK(RandomWrite)->RangeMultiplier(2)->Range(8192, 1<<30);

// g++
// ---------------------------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------
// RandomWrite/8192              248 ns          248 ns      2818836 bytes_per_second=30.7642Gi/s items_per_second=4.03232M/s
// RandomWrite/16384             496 ns          496 ns      1410703 bytes_per_second=30.762Gi/s items_per_second=2.01602M/s
// RandomWrite/32768            1010 ns         1009 ns       693609 bytes_per_second=30.2309Gi/s items_per_second=990.605k/s
// RandomWrite/65536            2231 ns         2231 ns       313471 bytes_per_second=27.3635Gi/s items_per_second=448.323k/s
// RandomWrite/131072           4539 ns         4539 ns       154093 bytes_per_second=26.8939Gi/s items_per_second=220.315k/s
// RandomWrite/262144           9443 ns         9442 ns        75904 bytes_per_second=25.8559Gi/s items_per_second=105.906k/s
// RandomWrite/524288          22958 ns        22956 ns        30774 bytes_per_second=21.2707Gi/s items_per_second=43.5624k/s
// RandomWrite/1048576         52714 ns        52710 ns        12910 bytes_per_second=18.5271Gi/s items_per_second=18.9718k/s
// RandomWrite/2097152        110205 ns       110189 ns         6281 bytes_per_second=17.7252Gi/s items_per_second=9.07531k/s
// RandomWrite/4194304       1311767 ns      1311596 ns          514 bytes_per_second=2.97824Gi/s items_per_second=762.43/s
// RandomWrite/8388608       5374467 ns      5373339 ns          129 bytes_per_second=1.45394Gi/s items_per_second=186.104/s
// RandomWrite/16777216     11947406 ns     11944769 ns           57 bytes_per_second=1.3081Gi/s items_per_second=83.7187/s
// RandomWrite/33554432     25559670 ns     25556076 ns           26 bytes_per_second=1.2228Gi/s items_per_second=39.1296/s
// RandomWrite/67108864     53841824 ns     53833641 ns           13 bytes_per_second=1.16098Gi/s items_per_second=18.5757/s
// RandomWrite/134217728   114749641 ns    114730479 ns            6 bytes_per_second=1.08951Gi/s items_per_second=8.71608/s
// RandomWrite/268435456   312067950 ns    311955452 ns            2 bytes_per_second=820.63Mi/s items_per_second=3.20559/s
// RandomWrite/536870912   742595377 ns    742485532 ns            1 bytes_per_second=689.576Mi/s items_per_second=1.34683/s
// RandomWrite/1073741824 1780222162 ns   1779682254 ns            1 bytes_per_second=575.384Mi/s items_per_second=0.561898/s

// ---------------------------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations UserCounters...
// ---------------------------------------------------------------------------------
// RandomWrite/8192              124 ns          124 ns      5651317 bytes_per_second=61.6726Gi/s items_per_second=8.08355M/s
// RandomWrite/16384             248 ns          248 ns      2831283 bytes_per_second=61.5545Gi/s items_per_second=4.03404M/s
// RandomWrite/32768             513 ns          513 ns      1351216 bytes_per_second=59.4371Gi/s items_per_second=1.94764M/s
// RandomWrite/65536            1372 ns         1372 ns       499221 bytes_per_second=44.4815Gi/s items_per_second=728.785k/s
// RandomWrite/131072           2997 ns         2996 ns       233136 bytes_per_second=40.7386Gi/s items_per_second=333.731k/s
// RandomWrite/262144           6006 ns         6005 ns       119369 bytes_per_second=40.6533Gi/s items_per_second=166.516k/s
// RandomWrite/524288          14143 ns        14141 ns        49447 bytes_per_second=34.529Gi/s items_per_second=70.7154k/s
// RandomWrite/1048576         35323 ns        35321 ns        19490 bytes_per_second=27.6484Gi/s items_per_second=28.312k/s
// RandomWrite/2097152         80003 ns        80003 ns         8454 bytes_per_second=24.4133Gi/s items_per_second=12.4996k/s
// RandomWrite/4194304       1139715 ns      1139644 ns          689 bytes_per_second=3.42761Gi/s items_per_second=877.467/s
// RandomWrite/8388608       4548121 ns      4546878 ns          156 bytes_per_second=1.71821Gi/s items_per_second=219.931/s
// RandomWrite/16777216     10303576 ns     10302102 ns           64 bytes_per_second=1.51668Gi/s items_per_second=97.0676/s
// RandomWrite/33554432     23753608 ns     23745833 ns           30 bytes_per_second=1.31602Gi/s items_per_second=42.1127/s
// RandomWrite/67108864     48998220 ns     48992297 ns           13 bytes_per_second=1.27571Gi/s items_per_second=20.4114/s
// RandomWrite/134217728   101894508 ns    101870060 ns            6 bytes_per_second=1.22705Gi/s items_per_second=9.81643/s
// RandomWrite/268435456   269753424 ns    269734496 ns            2 bytes_per_second=949.081Mi/s items_per_second=3.70735/s
// RandomWrite/536870912   662247757 ns    662104218 ns            1 bytes_per_second=773.292Mi/s items_per_second=1.51034/s
// RandomWrite/1073741824 1682667059 ns   1667864472 ns            1 bytes_per_second=613.959Mi/s items_per_second=0.599569/s
BENCHMARK_MAIN();