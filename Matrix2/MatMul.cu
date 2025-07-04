#include <benchmark/benchmark.h>
#include "Header.cuh"
#include "Tensor.cuh"

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

std::random_device rd;
std::mt19937 gen(rd());

int getRandomInt(int min, int max) {
  std::uniform_int_distribution<> dist(min, max);
  return dist(gen);
}

float randomFloatFromBits() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

  uint32_t bits = dist(gen);

  float result;
  std::memcpy(&result, &bits, sizeof(float));

  return result;
}

float randomFloat(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());  // Seeded Mersenne Twister
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}

void randTensor(Tensor a) {
  float* in = new float[a.row * a.col];
  for(int q = 0;q < a.row * a.col;q++) {
    in[q] = randomFloat(-1,1);
  }
  fromArray(in,a);
}

static void MatMulAB(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Tensor a(d1,d2);
  Tensor b(d2,d3);
  Tensor c(d1,d3);
  randTensor(a);
  randTensor(b);
  randTensor(c);

  for(auto _ : state) {
    MatMulPlusAsync(a,b,c,false,false);
    cudaDeviceSynchronize();
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatMulATB(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Tensor a(d2,d1);
  Tensor b(d2,d3);
  Tensor c(d1,d3);
  randTensor(a);
  randTensor(b);
  randTensor(c);

  for(auto _ : state) {
    MatMulPlusAsync(a,b,c,true,false);
    cudaDeviceSynchronize();
  }
  state.SetItemsProcessed(state.iterations());
}

static void MatMulABT(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int d3 = state.range(2);
  Tensor a(d1,d2);
  Tensor b(d3,d2);
  Tensor c(d1,d3);
  randTensor(a);
  randTensor(b);
  randTensor(c);

  for(auto _ : state) {
    MatMulPlusAsync(a,b,c,false,true);
    cudaDeviceSynchronize();
  }
  state.SetItemsProcessed(state.iterations());
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  b->Args({2800,512,512});
  b->Args({350,350,512});
}

#define BENCHMARK_TEMP(func) \
BENCHMARK_TEMPLATE(func,2800,512,512);\
BENCHMARK_TEMPLATE(func,350,350,512);\

//BENCHMARK(MatMulAB)->Apply(CustomArgs);
BENCHMARK(MatMulATB)->Apply(CustomArgs);
// BENCHMARK(MatMulABT)->Apply(CustomArgs);

BENCHMARK_MAIN();
