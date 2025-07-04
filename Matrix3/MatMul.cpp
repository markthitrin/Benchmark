#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>
#include <numeric>
#include <string>
#include <random>
#include <cmath>
#include <set>
#include <map>
#include <immintrin.h>
#include <cstdlib>
#include <memory>
#include <malloc.h>
#include <stdio.h>
#include <cstring>
#include "Matrix.h"

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
void randMatrix(float* data, int size) {
    for(int i  =0;i < size;i++) {
        data[i] = randomFloat(-1,1);
    }
}

template<int d1,int d2,int d3>
static void MatrixMulAB(benchmark::State& state) { // call operator*
    Matrix<d1, d2> A;
    Matrix<d2, d3> B;
    Matrix<d1, d3> C;
    randMatrix(A.data, d1 * d2);
    randMatrix(B.data, d2 * d3);
    randMatrix(C.data, d1 * d3);
    for(auto _ : state) {
        MatMulABPlus(A,B,C);
        escape(C.data);
    }
    state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2,int d3>
static void MatrixMulABT(benchmark::State& state) { // call operator*
    Matrix<d1, d2> A;
    Matrix<d3, d2> B;
    Matrix<d1, d3> C;
    randMatrix(A.data, d1 * d2);
    randMatrix(B.data, d3 * d2);
    randMatrix(C.data, d1 * d3);
    for(auto _ : state) {
        MatMulABTPlus(A,B,C);
        escape(C.data);
    }
    state.SetItemsProcessed(state.iterations());
}
template<int d1,int d2,int d3>
static void MatrixMulATB(benchmark::State& state) { // call operator*
    Matrix<d2, d1> A;
    Matrix<d2, d3> B;
    Matrix<d1, d3> C;
    randMatrix(A.data, d2 * d1);
    randMatrix(B.data, d2 * d3);
    randMatrix(C.data, d1 * d3);
    for(auto _ : state) {
        MatMulATBPlus(A,B,C);
        escape(C.data);
    }
    state.SetItemsProcessed(state.iterations());
}

#define BENCHMARK_TEMPAB(func) \
BENCHMARK_TEMPLATE(func,280,2048,512)->Iterations(12);;\
BENCHMARK_TEMPLATE(func,280,512,22465)->Iterations(1);;\
BENCHMARK_TEMPLATE(func,280,512,2048)->Iterations(12);;\
BENCHMARK_TEMPLATE(func,512,350,512)->Iterations(576);;\
BENCHMARK_TEMPLATE(func,64,350,350)->Iterations(2304);;\

#define BENCHMARK_TEMPABT(func) \
BENCHMARK_TEMPLATE(func,280,22465,512)->Iterations(1);;\
BENCHMARK_TEMPLATE(func,280,2048,512)->Iterations(12);;\
BENCHMARK_TEMPLATE(func,280,512,2048)->Iterations(12);;\
BENCHMARK_TEMPLATE(func,512,512,350)->Iterations(576);;\
BENCHMARK_TEMPLATE(func,64,350,350)->Iterations(2304);;\

#define BENCHMARK_TEMPATB(func) \
BENCHMARK_TEMPLATE(func,2048,280,512)->Iterations(12);;\
BENCHMARK_TEMPLATE(func,350,512,512)->Iterations(576);;\
BENCHMARK_TEMPLATE(func,350,64,350)->Iterations(2304);;\
BENCHMARK_TEMPLATE(func,512,280,2048)->Iterations(1);;\
BENCHMARK_TEMPLATE(func,512,280,22465)->Iterations(12);;\

BENCHMARK_TEMPAB(MatrixMulAB)
BENCHMARK_TEMPABT(MatrixMulABT)
BENCHMARK_TEMPATB(MatrixMulATB)

BENCHMARK_MAIN();