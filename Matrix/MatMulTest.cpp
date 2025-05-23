
#include <stdlib.h>
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
#include "Header.h"


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

bool check(const int size, float* a, float* b) {
  for(int q = 0;q < size;q++) {
    if(a[q] != b[q] && !(std::isnan(a[q]) && std::isnan(b[q]))) {
      std::cout << a[q] << " : " << b[q] << std::endl;
      return false;
    }
  }
  return true;
}

bool check(const int size, const Matrix& a, float* b) {
  for(int q = 0;q < size;q++) {
    if(a.data[q] != b[q] && !(std::isnan(a.data[q]) && std::isnan(b[q]))) {
      std::cout << a.data[q] << " : " << b[q] << std::endl;
      return false;
    }
  }
  return true;
}

template<int d1,int d2>
void print(Tensor2& a) {
  constexpr int fd2 = getColSizeFloat(d2);
  float* out = toFloat<d1,fd2>(a);
  for(int q = 0;q < d1;q++) {
    for(int w =0 ;w < fd2;w++) {
      std::cout << out[q * fd2 + w] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<int d1,int d2,int d3>
void MatMulATb1() {
  float* in1 = (float*)calloc(sizeof(float), d2 * d1);
  float* in2 = (float*)calloc(sizeof(float), d2 * d3);
  float* outans = (float*)calloc(sizeof(float), d1 * d3);
  float* out = (float*)calloc(sizeof(float), d1 * d3);
  for(int q = 0;q < d1 * d2;q++) {
    in1[q] = getRandomInt(-1,1);
  }
  for(int q = 0;q < d3 * d2;q++) {
    in2[q] = getRandomInt(-1,1);
  }
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d1;w++) {
      std::cout << in1[q * d1 + w] << " ";
    }
     std::cout << std::endl;
  }
  std::cout << std::endl;
  for(int q = 0;q < d2;q++) {
    for(int w = 0;w < d3;w++) {
      std::cout << in2[q * d3 + w] << " ";
    }
     std::cout << std::endl;
  }
  std::cout << std::endl;
  for(int q = 0;q < d1;q++){
    for(int w = 0;w < d3;w++) {
      for(int e = 0;e < d2;e++) {
        outans[q * d3 + w] += in1[e * d1 + q] * in2[e * d3 + w];
      }
    }
  }
  const int fd1 = getColSizeFloat(d1);
  Tensor2 a = create0<d2,d1>();
  Tensor2 b = create0<d2,d3>();
  Tensor2 c = create0<fd1,d3>();
  fromFloat<d2,d1>(in1,a);
  fromFloat<d2,d3>(in2,b);
  print<d2,d1>(a);
  print<d2,d3>(b);
  MatMulaTb2<d1,d2,d3>(a,b,c);
  print<fd1,d3>(c);
  out = toFloat<d1,d3>(c);
  for(int q = 0;q < d1;q++) {
    for(int w = 0;w < d3;w++) {
      std::cout << outans[q * d3 + w] << " ";
    }
     std::cout << std::endl;
  }
  std::cout << std::endl;
  for(int q = 0;q < d1;q++) {
    for(int w=0;w < d3;w++) {
      std::cout << out[q * d3 + w] << " ";
    }
     std::cout << std::endl;
  }
  std::cout << std::endl;
  if (!check(d1 * d3, out, outans)) {
    std::cout << "MatMulATb1 failed" << std::endl;
  }
  else {
    std::cout << "MatMulATb1 passed" << std::endl;
  }
}

int main() {
  MatMulATb1<8,9,4>();
}