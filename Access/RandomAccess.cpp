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
BENCHMARK(RandomAccess)->RangeMultiplier(2)->Range(1024, 1 << 30);

//g++
// ----------------------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------
// RandomAccess/1024             9.23 ns         9.23 ns     76440231 bytes_per_second=103.33Gi/s items_per_second=108.35M/s
// RandomAccess/2048             18.3 ns         18.2 ns     38677011 bytes_per_second=104.535Gi/s items_per_second=54.8064M/s
// RandomAccess/4096             36.1 ns         36.1 ns     19369541 bytes_per_second=105.642Gi/s items_per_second=27.6934M/s
// RandomAccess/8192             72.8 ns         72.8 ns      9532719 bytes_per_second=104.854Gi/s items_per_second=13.7434M/s
// RandomAccess/16384             169 ns          169 ns      4132266 bytes_per_second=90.209Gi/s items_per_second=5.91193M/s
// RandomAccess/32768             424 ns          424 ns      1649898 bytes_per_second=71.9841Gi/s items_per_second=2.35878M/s  L1 1 instance
// RandomAccess/65536             975 ns          975 ns       713783 bytes_per_second=62.6219Gi/s items_per_second=1.026M/s
// RandomAccess/131072           2098 ns         2098 ns       332149 bytes_per_second=58.192Gi/s items_per_second=476.709k/s
// RandomAccess/262144           4541 ns         4541 ns       154271 bytes_per_second=53.7597Gi/s items_per_second=220.2k/s    L1 MAX
// RandomAccess/524288           9137 ns         9136 ns        76542 bytes_per_second=53.4448Gi/s items_per_second=109.455k/s  L2 1 instance
// RandomAccess/1048576         18446 ns        18444 ns        37905 bytes_per_second=52.9474Gi/s items_per_second=54.2182k/s
// RandomAccess/2097152         36984 ns        36986 ns        18946 bytes_per_second=52.8079Gi/s items_per_second=27.0376k/s  L3 1 instance
// RandomAccess/4194304         74461 ns        74459 ns         9360 bytes_per_second=52.4618Gi/s items_per_second=13.4302k/s  L2 MAX
// RandomAccess/8388608        152656 ns       152642 ns         4553 bytes_per_second=51.1817Gi/s items_per_second=6.55126k/s  L3 MAX
// RandomAccess/16777216       838774 ns       838719 ns          835 bytes_per_second=18.6296Gi/s items_per_second=1.19229k/s
// RandomAccess/33554432      2845200 ns      2844726 ns          243 bytes_per_second=10.9852Gi/s items_per_second=351.528/s
// RandomAccess/67108864      7164445 ns      7162434 ns           86 bytes_per_second=8.72608Gi/s items_per_second=139.617/s
// RandomAccess/134217728    16194493 ns     16192671 ns           39 bytes_per_second=7.71954Gi/s items_per_second=61.7563/s
// RandomAccess/268435456    39828373 ns     39822909 ns           17 bytes_per_second=6.27779Gi/s items_per_second=25.1112/s
// RandomAccess/536870912   110693961 ns    110678012 ns            6 bytes_per_second=4.51761Gi/s items_per_second=9.03522/s
// RandomAccess/1073741824  314705703 ns    314655471 ns            2 bytes_per_second=3.17808Gi/s items_per_second=3.17808/s
//   0.59 │       shl          $0x5,%rdx                                                                                                                                     ◆
//   0.50 │       add          %r12,%rdx                                                                                                                                     ▒
//   0.60 │       vmovdqa      (%rdx),%ymm0                                                                                                                                  ▒
//   0.76 │       movslq       0x4(%rax),%rdx                                                                                                                           ▒
//   0.46 │       vmovdqa      %ymm0,0x80(%rsp)


// ----------------------------------------------------------------------------------
// Benchmark                        Time             CPU   Iterations UserCounters...
// ----------------------------------------------------------------------------------
// RandomAccess/1024             10.7 ns         10.7 ns     64941551 bytes_per_second=89.3158Gi/s items_per_second=93.6544M/s
// RandomAccess/2048             21.3 ns         21.3 ns     33089846 bytes_per_second=89.4487Gi/s items_per_second=46.8969M/s
// RandomAccess/4096             42.4 ns         42.4 ns     16496019 bytes_per_second=89.9181Gi/s items_per_second=23.5715M/s
// RandomAccess/8192             85.0 ns         85.0 ns      8236829 bytes_per_second=89.7308Gi/s items_per_second=11.7612M/s
// RandomAccess/16384             170 ns          170 ns      4106646 bytes_per_second=89.5601Gi/s items_per_second=5.86941M/s
// RandomAccess/32768             413 ns          413 ns      1695892 bytes_per_second=73.9533Gi/s items_per_second=2.4233M/s  L1 1 instance
// RandomAccess/65536             954 ns          954 ns       731857 bytes_per_second=64.001Gi/s items_per_second=1.04859M/s
// RandomAccess/131072           2062 ns         2062 ns       338725 bytes_per_second=59.2055Gi/s items_per_second=485.011k/s
// RandomAccess/262144           4461 ns         4461 ns       156926 bytes_per_second=54.7315Gi/s items_per_second=224.18k/s  L1 MAX
// RandomAccess/524288           8949 ns         8948 ns        78195 bytes_per_second=54.5704Gi/s items_per_second=111.76k/s  L2 1 instance
// RandomAccess/1048576         18053 ns        18049 ns        38757 bytes_per_second=54.1051Gi/s items_per_second=55.4037k/s
// RandomAccess/2097152         36116 ns        36111 ns        19400 bytes_per_second=54.0875Gi/s items_per_second=27.6928k/s L3 1 instance
// RandomAccess/4194304         73041 ns        73025 ns         9577 bytes_per_second=53.492Gi/s items_per_second=13.694k/s   L2 MAX
// RandomAccess/8388608        148475 ns       148459 ns         4672 bytes_per_second=52.624Gi/s items_per_second=6.73588k/s  L3 MAX
// RandomAccess/16777216       829247 ns       829085 ns          848 bytes_per_second=18.8461Gi/s items_per_second=1.20615k/s
// RandomAccess/33554432      2829220 ns      2828594 ns          243 bytes_per_second=11.0479Gi/s items_per_second=353.533/s
// RandomAccess/67108864      7113589 ns      7112485 ns           86 bytes_per_second=8.78736Gi/s items_per_second=140.598/s
// RandomAccess/134217728    16505643 ns     16502203 ns           39 bytes_per_second=7.57475Gi/s items_per_second=60.598/s
// RandomAccess/268435456    41368069 ns     41364527 ns           16 bytes_per_second=6.04383Gi/s items_per_second=24.1753/s
// RandomAccess/536870912   125278318 ns    125267384 ns            6 bytes_per_second=3.99146Gi/s items_per_second=7.98292/s
// RandomAccess/1073741824  314317198 ns    314279603 ns            2 bytes_per_second=3.18188Gi/s items_per_second=3.18188/s
//  0.96 │       shl          $0x5,%rcx                                                                                                                                     ▒
//   1.08 │       vmovaps      (%r12,%rcx,1),%ymm0                                                                                                                           ▒
//   0.97 │       movslq       0x58(%rax),%rcx                                                                                                                               ▒
//   0.94 │       shl          $0x5,%rcx                                                                                                                                     ▒
//   1.14 │       vmovaps      (%r12,%rcx,1),%ymm0                                                                                                                           ▒
//   1.06 │       movslq       0x5c(%rax),%rcx   


// The <=8192 bytes version of alng version is always little slower, As it might occus some stall with repeated instruction length that is
// not compatible with 4-way superscalar. Inserting "u = 1" with u being volatile int increase the speed of clang++ <=8192 bytes version
// as fast as g++ while making g++ fastest memory request cap at 61 GB/s.


// Note1 : L1d 256 KB (8 instances)   32  KB each
//         L2    4 MB (8 instances)   512 KB each
//         L3    8 MB (2 instances)   2   MB each
BENCHMARK_MAIN();