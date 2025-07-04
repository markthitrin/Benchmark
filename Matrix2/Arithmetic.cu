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

static void Plus1(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  Tensor a(d1,d2);
  Tensor b(d1,d2);
  Tensor c(d1,d2);
  randTensor(a);
  randTensor(b);
  randTensor(c);

  for(auto _ : state) {
    plusAsync(a,b,c);
    cudaDeviceSynchronize();
  }
  state.SetItemsProcessed(state.iterations());
}

static void Plus2(benchmark::State& state) {
  const int d1 = state.range(0);
  const int d2 = state.range(1);
  const int batch = state.range(2);
  Tensor a(d1 * batch,d2);
  Tensor b(d1,d2);
  Tensor c(d1 * batch,d2);
  randTensor(a);
  randTensor(b);
  randTensor(c);

  for(auto _ : state) {
    PlusBatch(a,b,c, batch);
    cudaDeviceSynchronize();
  }
  state.SetItemsProcessed(state.iterations());
}

static void CustomArgs(benchmark::internal::Benchmark* b) {
  // b->Args({1, 8});
  // b->Args({1, 32});
  // b->Args({1, 64});
  // b->Args({1, 128});
  // b->Args({1, 512});
  // b->Args({1, 2048});
  // b->Args({8, 8});
  // b->Args({8, 32});
  // b->Args({8, 64});
  // b->Args({8, 128});
  // b->Args({8, 512});
  // b->Args({8, 2048});
  // b->Args({32, 8});
  // b->Args({32, 32});
  // b->Args({32, 64});
  // b->Args({32, 128});
  // b->Args({32, 512});
  // b->Args({32, 2048});
  // b->Args({1024, 8});
  // b->Args({1024, 32});
  // b->Args({1024, 64});
  // b->Args({1024, 128});
  // b->Args({1024, 512});
  // b->Args({1024, 2048, 1});
  // b->Args({1024, 2048, 2});
  // b->Args({1024, 2048, 4});
  // b->Args({1024, 2048, 8});
  // b->Args({1024, 2048, 16});
  // b->Args({1024, 2048, 32});
  b->Args({1024, 2048, 64});
  // b->Args({1024, 2048, 128});
  // b->Args({1024, 2048, 256});
  // b->Args({1024, 2048, 512});
}

BENCHMARK(Plus2)->Apply(CustomArgs);

BENCHMARK_MAIN();