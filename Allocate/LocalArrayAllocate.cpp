#include <benchmark/benchmark.h>
#define BENCHMARK_TEMPLATE_RANGE(func) \
    BENCHMARK(func<1>);\
    BENCHMARK(func<4>);\
    BENCHMARK(func<16>);\
    BENCHMARK(func<64>);\
    BENCHMARK(func<256>);\
    BENCHMARK(func<1024>);\
    BENCHMARK(func<4096>);\
    BENCHMARK(func<16384>);\
    BENCHMARK(func<65536>);\
    BENCHMARK(func<262144>);\
    BENCHMARK(func<1048576>);

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

template<int array_size>
static void CharAllocate(benchmark::State& state) {
  for(auto _ : state) {
    char c[array_size];
    escape(&c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(char) * array_size);
}

template<int array_size>
static void ShortAllocate(benchmark::State& state) {
  for(auto _ : state) {
    short s[array_size];
    escape(&s);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(short) * array_size);
}

template<int array_size>
static void IntAllocate(benchmark::State& state) {
  for(auto _ : state) {
    int i[array_size];
    escape(&i);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(int) * array_size);
}

template<int array_size>
static void LongLongAllocate(benchmark::State& state) {
  for(auto _ : state) {
    long long l[array_size];
    escape(&l);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long long) * array_size);
}

template<int array_size>
static void FloatAllocate(benchmark::State& state) {
  for(auto _ : state) {
    float f[array_size];
    escape(&f);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) * array_size);
}

template<int array_size>
static void DoubleAllocate(benchmark::State& state) {
  for(auto _ : state) {
    double d[array_size];
    escape(&d);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(double) * array_size);

}

template<int array_size>
static void LongDoubleAllocate(benchmark::State& state) {
  for(auto _ : state) {
    long double ld[array_size];
    escape(&ld);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(long double) * array_size);
}

BENCHMARK_TEMPLATE_RANGE(CharAllocate);
BENCHMARK_TEMPLATE_RANGE(ShortAllocate);
BENCHMARK_TEMPLATE_RANGE(IntAllocate);
BENCHMARK_TEMPLATE_RANGE(LongLongAllocate);
BENCHMARK_TEMPLATE_RANGE(FloatAllocate);
BENCHMARK_TEMPLATE_RANGE(DoubleAllocate); 
BENCHMARK_TEMPLATE_RANGE(LongDoubleAllocate);

// g++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// CharAllocate<1>                  0.119 ns        0.119 ns   5894975123 bytes_per_second=7.85876Gi/s items_per_second=8.43828G/s
// CharAllocate<4>                  0.119 ns        0.119 ns   5837367830 bytes_per_second=31.2891Gi/s items_per_second=8.39911G/s
// CharAllocate<16>                 0.120 ns        0.120 ns   5655999782 bytes_per_second=124.212Gi/s items_per_second=8.33575G/s
// CharAllocate<64>                 0.120 ns        0.120 ns   5819425053 bytes_per_second=496.864Gi/s items_per_second=8.33599G/s
// CharAllocate<256>                0.119 ns        0.119 ns   5856658452 bytes_per_second=1.94894Ti/s items_per_second=8.37062G/s
// CharAllocate<1024>               0.120 ns        0.120 ns   5853571618 bytes_per_second=7.7782Ti/s items_per_second=8.35178G/s
// CharAllocate<4096>               0.120 ns        0.120 ns   5842537131 bytes_per_second=31.1279Ti/s items_per_second=8.35584G/s
// CharAllocate<16384>              0.120 ns        0.120 ns   5843144509 bytes_per_second=124.266Ti/s items_per_second=8.33934G/s
// CharAllocate<65536>              0.120 ns        0.120 ns   5832774359 bytes_per_second=496.903Ti/s items_per_second=8.33666G/s
// CharAllocate<262144>             0.119 ns        0.119 ns   5838095078 bytes_per_second=1.94916Pi/s items_per_second=8.37157G/s
// CharAllocate<1048576>            0.120 ns        0.120 ns   5853620714 bytes_per_second=7.79205Pi/s items_per_second=8.36665G/s
// ShortAllocate<1>                 0.120 ns        0.120 ns   5827080876 bytes_per_second=15.5787Gi/s items_per_second=8.36374G/s
// ShortAllocate<4>                 0.120 ns        0.119 ns   5855236602 bytes_per_second=62.3514Gi/s items_per_second=8.36866G/s
// ShortAllocate<16>                0.120 ns        0.120 ns   5852372320 bytes_per_second=249.365Gi/s items_per_second=8.36729G/s
// ShortAllocate<64>                0.120 ns        0.120 ns   5854180313 bytes_per_second=995.004Gi/s items_per_second=8.3467G/s
// ShortAllocate<256>               0.120 ns        0.120 ns   5807493761 bytes_per_second=3.8873Ti/s items_per_second=8.34791G/s
// ShortAllocate<1024>              0.119 ns        0.119 ns   5839462238 bytes_per_second=15.591Ti/s items_per_second=8.37034G/s
// ShortAllocate<4096>              0.120 ns        0.119 ns   5826083792 bytes_per_second=62.3484Ti/s items_per_second=8.36826G/s
// ShortAllocate<16384>             0.120 ns        0.120 ns   5832562560 bytes_per_second=249.028Ti/s items_per_second=8.356G/s
// ShortAllocate<65536>             0.120 ns        0.120 ns   5844172325 bytes_per_second=995.627Ti/s items_per_second=8.35193G/s
// ShortAllocate<262144>            0.120 ns        0.120 ns   5846986925 bytes_per_second=3.89178Pi/s items_per_second=8.35753G/s
// ShortAllocate<1048576>           0.120 ns        0.120 ns   5835673001 bytes_per_second=15.5382Pi/s items_per_second=8.34202G/s
// IntAllocate<1>                   0.120 ns        0.120 ns   5795392109 bytes_per_second=30.9759Gi/s items_per_second=8.31504G/s
// IntAllocate<4>                   0.120 ns        0.120 ns   5783873646 bytes_per_second=124.061Gi/s items_per_second=8.32561G/s
// IntAllocate<16>                  0.121 ns        0.121 ns   5792325839 bytes_per_second=494.559Gi/s items_per_second=8.29732G/s
// IntAllocate<64>                  0.120 ns        0.120 ns   5817711179 bytes_per_second=1.94032Ti/s items_per_second=8.3336G/s
// IntAllocate<256>                 0.120 ns        0.120 ns   5820874189 bytes_per_second=7.76791Ti/s items_per_second=8.34073G/s
// IntAllocate<1024>                0.120 ns        0.120 ns   5846173772 bytes_per_second=31.0829Ti/s items_per_second=8.34376G/s
// IntAllocate<4096>                0.120 ns        0.120 ns   5813664688 bytes_per_second=123.991Ti/s items_per_second=8.32087G/s
// IntAllocate<16384>               0.120 ns        0.120 ns   5838403354 bytes_per_second=496.702Ti/s items_per_second=8.33327G/s
// IntAllocate<65536>               0.120 ns        0.120 ns   5829721384 bytes_per_second=1.94741Pi/s items_per_second=8.36408G/s
// IntAllocate<262144>              0.119 ns        0.119 ns   5871447640 bytes_per_second=7.82323Pi/s items_per_second=8.40013G/s
// IntAllocate<1048576>             0.120 ns        0.120 ns   5832818636 bytes_per_second=31.0406Pi/s items_per_second=8.3324G/s
// LongLongAllocate<1>              0.120 ns        0.120 ns   5822857818 bytes_per_second=61.8706Gi/s items_per_second=8.30413G/s
// LongLongAllocate<4>              0.120 ns        0.120 ns   5838478931 bytes_per_second=249.38Gi/s items_per_second=8.36782G/s
// LongLongAllocate<16>             0.119 ns        0.119 ns   5848164473 bytes_per_second=997.761Gi/s items_per_second=8.36982G/s
// LongLongAllocate<64>             0.119 ns        0.119 ns   5852571859 bytes_per_second=3.9035Ti/s items_per_second=8.38269G/s
// LongLongAllocate<256>            0.119 ns        0.119 ns   5872662702 bytes_per_second=15.6668Ti/s items_per_second=8.41107G/s
// LongLongAllocate<1024>           0.119 ns        0.119 ns   5872408437 bytes_per_second=62.6212Ti/s items_per_second=8.40488G/s
// LongLongAllocate<4096>           0.119 ns        0.119 ns   5869176836 bytes_per_second=250.577Ti/s items_per_second=8.40798G/s
// LongLongAllocate<16384>          0.119 ns        0.119 ns   5874392793 bytes_per_second=1002.78Ti/s items_per_second=8.41191G/s
// LongLongAllocate<65536>          0.119 ns        0.119 ns   5869372110 bytes_per_second=3.91685Pi/s items_per_second=8.41137G/s
// LongLongAllocate<262144>         0.119 ns        0.119 ns   5874476749 bytes_per_second=15.6666Pi/s items_per_second=8.41093G/s
//  49.92 │188:┌─→sub      $0x2,%rax                                                                                                                                         ▒
//  16.67 │    │↑ je       37                                                                                                                                                ▒
//  16.40 │    │  sub      $0x2,%rax                                                                                                                                         ▒
//  17.00 │    └──jne      188


// clang++
// --------------------------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations UserCounters...
// --------------------------------------------------------------------------------------
// CharAllocate<1>                  0.237 ns        0.237 ns   2949892846 bytes_per_second=3.92622Gi/s items_per_second=4.21574G/s
// CharAllocate<4>                  0.237 ns        0.237 ns   2953066322 bytes_per_second=15.708Gi/s items_per_second=4.21658G/s
// CharAllocate<16>                 0.476 ns        0.476 ns   1470768043 bytes_per_second=31.2834Gi/s items_per_second=2.0994G/s
// CharAllocate<64>                 0.237 ns        0.237 ns   2953163560 bytes_per_second=251.602Gi/s items_per_second=4.22118G/s
// CharAllocate<256>                0.237 ns        0.237 ns   2951469074 bytes_per_second=1006.06Gi/s items_per_second=4.21973G/s
// CharAllocate<1024>               0.476 ns        0.476 ns   1468778493 bytes_per_second=1.95667Ti/s items_per_second=2.10096G/s
// CharAllocate<4096>               0.238 ns        0.238 ns   2953319353 bytes_per_second=15.6266Ti/s items_per_second=4.19474G/s
// CharAllocate<16384>              0.239 ns        0.239 ns   2934266015 bytes_per_second=62.4758Ti/s items_per_second=4.19268G/s
// CharAllocate<65536>              0.238 ns        0.238 ns   2934283137 bytes_per_second=249.982Ti/s items_per_second=4.194G/s
// CharAllocate<262144>             0.478 ns        0.478 ns   1464422025 bytes_per_second=499.254Ti/s items_per_second=2.09402G/s
// CharAllocate<1048576>            0.239 ns        0.239 ns   2936657017 bytes_per_second=3.89257Pi/s items_per_second=4.17962G/s
// ShortAllocate<1>                 0.238 ns        0.238 ns   2937997101 bytes_per_second=7.81743Gi/s items_per_second=4.19695G/s
// ShortAllocate<4>                 0.477 ns        0.477 ns   1319657036 bytes_per_second=15.6101Gi/s items_per_second=2.09515G/s
// ShortAllocate<16>                0.238 ns        0.238 ns   2936730828 bytes_per_second=125.08Gi/s items_per_second=4.19697G/s
// ShortAllocate<64>                0.238 ns        0.238 ns   2937506611 bytes_per_second=500.332Gi/s items_per_second=4.19709G/s
// ShortAllocate<256>               0.477 ns        0.477 ns   1465892167 bytes_per_second=999.436Gi/s items_per_second=2.09597G/s
// ShortAllocate<1024>              0.238 ns        0.238 ns   2934113468 bytes_per_second=7.81438Ti/s items_per_second=4.19532G/s
// ShortAllocate<4096>              0.238 ns        0.238 ns   2937856767 bytes_per_second=31.2584Ti/s items_per_second=4.19544G/s
// ShortAllocate<16384>             0.237 ns        0.237 ns   2954307997 bytes_per_second=125.772Ti/s items_per_second=4.22021G/s
// ShortAllocate<65536>             0.476 ns        0.476 ns   1470192219 bytes_per_second=250.336Ti/s items_per_second=2.09997G/s
// ShortAllocate<262144>            0.237 ns        0.237 ns   2952373384 bytes_per_second=1.96517Pi/s items_per_second=4.22017G/s
// ShortAllocate<1048576>           0.237 ns        0.237 ns   2953615984 bytes_per_second=7.86305Pi/s items_per_second=4.22144G/s
// IntAllocate<1>                   0.237 ns        0.237 ns   2951657720 bytes_per_second=15.7242Gi/s items_per_second=4.22094G/s
// IntAllocate<4>                   0.237 ns        0.237 ns   2954397075 bytes_per_second=62.8981Gi/s items_per_second=4.22102G/s
// IntAllocate<16>                  0.476 ns        0.476 ns   1470695105 bytes_per_second=125.197Gi/s items_per_second=2.10046G/s
// IntAllocate<64>                  0.476 ns        0.476 ns   1468808755 bytes_per_second=500.837Gi/s items_per_second=2.10066G/s
// IntAllocate<256>                 0.238 ns        0.238 ns   2954132252 bytes_per_second=3.91412Ti/s items_per_second=4.20276G/s
// IntAllocate<1024>                0.239 ns        0.239 ns   2934192205 bytes_per_second=15.6175Ti/s items_per_second=4.19229G/s
// IntAllocate<4096>                0.239 ns        0.239 ns   2934303850 bytes_per_second=62.473Ti/s items_per_second=4.19249G/s
// IntAllocate<16384>               0.478 ns        0.478 ns   1465650410 bytes_per_second=124.646Ti/s items_per_second=2.09122G/s
// IntAllocate<65536>               0.239 ns        0.239 ns   2934767752 bytes_per_second=999.637Ti/s items_per_second=4.19278G/s
// IntAllocate<262144>              0.239 ns        0.238 ns   2933915695 bytes_per_second=3.90494Pi/s items_per_second=4.19289G/s
// IntAllocate<1048576>             0.238 ns        0.238 ns   2937737935 bytes_per_second=15.6327Pi/s items_per_second=4.19636G/s
// LongLongAllocate<1>              0.237 ns        0.237 ns   2950229025 bytes_per_second=31.4477Gi/s items_per_second=4.22084G/s
// LongLongAllocate<4>              0.237 ns        0.237 ns   2953276528 bytes_per_second=125.731Gi/s items_per_second=4.21883G/s
// LongLongAllocate<16>             0.237 ns        0.237 ns   2858376322 bytes_per_second=503.084Gi/s items_per_second=4.22017G/s
// LongLongAllocate<64>             0.237 ns        0.237 ns   2953525832 bytes_per_second=1.96588Ti/s items_per_second=4.22169G/s
// LongLongAllocate<256>            0.476 ns        0.476 ns   1470283612 bytes_per_second=3.91351Ti/s items_per_second=2.10105G/s
// LongLongAllocate<1024>           0.240 ns        0.240 ns   2952268354 bytes_per_second=31.0595Ti/s items_per_second=4.16873G/s
// LongLongAllocate<4096>           0.237 ns        0.237 ns   2954114150 bytes_per_second=125.752Ti/s items_per_second=4.21954G/s
// LongLongAllocate<16384>          0.237 ns        0.237 ns   2953363288 bytes_per_second=503.083Ti/s items_per_second=4.22017G/s
// LongLongAllocate<65536>          0.476 ns        0.476 ns   1470425793 bytes_per_second=1001.67Ti/s items_per_second=2.10066G/s
// LongLongAllocate<262144>         0.237 ns        0.237 ns   2954808267 bytes_per_second=7.86399Pi/s items_per_second=4.22195G/s
//  16.90 │ 40:┌─→mov      %rax,-0x30(%rbp)
//  16.37 │    │  test     %r14,%r14
//        │    │↓ jle      1dc
//  16.94 │    │  dec      %r14
//  49.80 │    └──jne      40

// Note1: The allocation of array on stack require no time as it is set at compile time.
// Note2: We can see the effect of the superscaler with clang too.

BENCHMARK_MAIN();