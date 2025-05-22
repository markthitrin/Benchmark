
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

int main() {
  const int batch = 16;
  const int d1 = 16;
  const int d2 = 256;
  const int d3 = 32;
  Tensor t(batch, d1, d2);
  Matrix m(d2, d3);
  Tensor out(batch, d1, d3);
  for(int q = 0;q < batch;q++) {
    for(int w = 0;w < d1;w++) {
      for(int e = 0;e < d2;e++) {
        t[q][w][e] = randomFloat(-1,1);
      }
    }
  }
  for(int w = 0;w < d2;w++) {
    for(int e = 0;e < d3;e++) {
      m[w][e] = randomFloat(-1,1);
    }
  }
  out = t * m;
  for(int q = 0;q < batch;q++) {
    for(int w = 0;w < d1;w++) {
      for(int e = 0;e < d2;e++) {
        std::cout << out[q][w][e] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}