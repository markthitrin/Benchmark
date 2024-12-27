#include <benchmark/benchmark.h>
#include <vector>

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void CharAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<char> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

static void ShortAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<short> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

static void IntAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<int> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

static void LongLongAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long long> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

static void FloatAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<float> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

static void DoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<double> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

static void LongDoubleAllocate(benchmark::State& state) {
  const int array_size = state.range(0);
  for(auto _ : state) {
    std::vector<long double> v;
    for(int i = 0;i < array_size;i++) {
        v.push_back(0);
    }
    escape(v.data());
  }
}

// BENCHMARK(CharAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// ---------------------------------------------------------------
// Benchmark                     Time             CPU   Iterations
// ---------------------------------------------------------------
// CharAllocate/1             13.0 ns         13.0 ns     53992999
// CharAllocate/4             50.2 ns         50.2 ns     13851570
// CharAllocate/16            92.1 ns         92.1 ns      7431317
// CharAllocate/64             176 ns          176 ns      3973496
// CharAllocate/256            471 ns          471 ns      1488029
// CharAllocate/1024          1594 ns         1593 ns       438969
// CharAllocate/4096          6037 ns         6037 ns       115718
// CharAllocate/16384        23692 ns        23689 ns        29540
// CharAllocate/65536        94488 ns        94478 ns         7405
// CharAllocate/262144      572818 ns       572702 ns         1239
// CharAllocate/1048576    2800096 ns      2799502 ns          248

// BENCHMARK(ShortAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// ShortAllocate/1             13.8 ns         13.8 ns     50940800
// ShortAllocate/4             54.9 ns         54.9 ns     13111281
// ShortAllocate/16            98.7 ns         98.7 ns      7177332
// ShortAllocate/64             182 ns          182 ns      3879369
// ShortAllocate/256            483 ns          483 ns      1454338
// ShortAllocate/1024          1626 ns         1626 ns       430644
// ShortAllocate/4096          6122 ns         6121 ns       114233
// ShortAllocate/16384        24188 ns        24184 ns        28951
// ShortAllocate/65536       160927 ns       160688 ns         4361
// ShortAllocate/262144      794245 ns       794126 ns          874
// ShortAllocate/1048576    3544731 ns      3544074 ns          198

// BENCHMARK(IntAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// --------------------------------------------------------------
// IntAllocate/1             13.7 ns         13.7 ns     51457665
// IntAllocate/4             56.2 ns         56.2 ns     12566971
// IntAllocate/16             102 ns          101 ns      6890830
// IntAllocate/64             167 ns          167 ns      4192416
// IntAllocate/256            425 ns          425 ns      1633824
// IntAllocate/1024          1382 ns         1382 ns       570840
// IntAllocate/4096          4927 ns         4927 ns       139975
// IntAllocate/16384        13317 ns        13314 ns        48087
// IntAllocate/65536       233403 ns       233259 ns         3001
// IntAllocate/262144     1113960 ns      1113762 ns          620
// IntAllocate/1048576    5271030 ns      5269226 ns          122

// BENCHMARK(LongLongAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// -------------------------------------------------------------------
// Benchmark                         Time             CPU   Iterations
// -------------------------------------------------------------------
// LongLongAllocate/1             13.5 ns         13.5 ns     51474467
// LongLongAllocate/4             53.7 ns         53.7 ns     12028687
// LongLongAllocate/16            97.1 ns         97.1 ns      7022748
// LongLongAllocate/64             155 ns          155 ns      4365350
// LongLongAllocate/256            358 ns          358 ns      1961411
// LongLongAllocate/1024          1026 ns         1026 ns       682944
// LongLongAllocate/4096          3744 ns         3744 ns       186821
// LongLongAllocate/16384        79302 ns        78954 ns         8738
// LongLongAllocate/65536       467154 ns       467066 ns         1498
// LongLongAllocate/262144     2228785 ns      2228074 ns          310
// LongLongAllocate/1048576   10149095 ns     10146107 ns           65

// BENCHMARK(FloatAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// ----------------------------------------------------------------
// Benchmark                      Time             CPU   Iterations
// ----------------------------------------------------------------
// FloatAllocate/1             13.7 ns         13.7 ns     51202305
// FloatAllocate/4             58.3 ns         58.3 ns     11740631
// FloatAllocate/16             104 ns          104 ns      6655529
// FloatAllocate/64             166 ns          166 ns      4207105
// FloatAllocate/256            343 ns          343 ns      2075694
// FloatAllocate/1024           987 ns          987 ns       700046
// FloatAllocate/4096          3451 ns         3450 ns       203584
// FloatAllocate/16384        13321 ns        13319 ns        52491
// FloatAllocate/65536       234746 ns       234590 ns         2983
// FloatAllocate/262144     1121043 ns      1120798 ns          616
// FloatAllocate/1048576    5295567 ns      5294331 ns          123

// BENCHMARK(DoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// -----------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations
// -----------------------------------------------------------------
// DoubleAllocate/1             13.6 ns         13.6 ns     51455441
// DoubleAllocate/4             54.9 ns         54.9 ns     12752502
// DoubleAllocate/16            99.9 ns         99.9 ns      7346521
// DoubleAllocate/64             163 ns          163 ns      4307235
// DoubleAllocate/256            362 ns          362 ns      1870403
// DoubleAllocate/1024          1030 ns         1030 ns       678684
// DoubleAllocate/4096          3796 ns         3796 ns       184525
// DoubleAllocate/16384        79406 ns        79044 ns         8722
// DoubleAllocate/65536       468182 ns       468138 ns         1496
// DoubleAllocate/262144     2224569 ns      2224019 ns          315
// DoubleAllocate/1048576   10313470 ns     10311453 ns           65

// BENCHMARK(LongDoubleAllocate)->RangeMultiplier(4)->Range(1, 1<<20);
// ---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
// ---------------------------------------------------------------------
// LongDoubleAllocate/1             17.3 ns         17.3 ns     40419606
// LongDoubleAllocate/4             56.4 ns         56.4 ns     12374312
// LongDoubleAllocate/16             119 ns          119 ns      5863922
// LongDoubleAllocate/64             283 ns          283 ns      2475897
// LongDoubleAllocate/256            887 ns          887 ns       786622
// LongDoubleAllocate/1024          3176 ns         3175 ns       220340
// LongDoubleAllocate/4096         12566 ns        12565 ns        55650
// LongDoubleAllocate/16384       230805 ns       230657 ns         3032
// LongDoubleAllocate/65536      1107078 ns      1106931 ns          626
// LongDoubleAllocate/262144     5252742 ns      5251396 ns          123
// LongDoubleAllocate/1048576   23208367 ns     23205423 ns           30

BENCHMARK_MAIN();