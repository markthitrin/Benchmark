
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

template<int size>
void Plus1() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float* in2 = (float*)malloc(sizeof(float) * size);
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      in2[q] = randomFloatFromBits();
      ans[q] = in1[q] + in2[q];
    }
    Tensor3 a = Create<1,size>();
    Tensor3 b = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    FromArray<1,size>(in2,b);
    Plus<1,size>(a,b,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;  
    }
  }
  if(pass) {
    std::cout << "Plus1 : passed\n";
  }
  else {
    std::cout << "Plus1 : filed\n";
  }
}
template<int size>
void Plus2() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float in2 = randomFloatFromBits();
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      ans[q] = in1[q] + in2;
    }
    Tensor3 a = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    Plus<1,size>(a,in2,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Plus2 : passed\n";
  }
  else {
    std::cout << "Plus2 : filed\n";
  }
}

template<int size>
void Sub1() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float* in2 = (float*)malloc(sizeof(float) * size);
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      in2[q] = randomFloatFromBits();
      ans[q] = in1[q] - in2[q];
    }
    Tensor3 a = Create<1,size>();
    Tensor3 b = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    FromArray<1,size>(in2,b);
    Sub<1,size>(a,b,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Sub1 : passed\n";
  }
  else {
    std::cout << "Sub1 : filed\n";
  }
}
template<int size>
void Sub2() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float in2 = randomFloatFromBits();
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      ans[q] = in1[q] - in2;
    }
    Tensor3 a = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    Sub<1,size>(a,in2,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Sub2 : passed\n";
  }
  else {
    std::cout << "Sub2 : filed\n";
  }
}

template<int size>
void Mul1() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float* in2 = (float*)malloc(sizeof(float) * size);
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      in2[q] = randomFloatFromBits();
      ans[q] = in1[q] * in2[q];
    }
    Tensor3 a = Create<1,size>();
    Tensor3 b = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    FromArray<1,size>(in2,b);
    Mul<1,size>(a,b,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Mul1 : passed\n";
  }
  else {
    std::cout << "Mul1 : filed\n";
  }
}
template<int size>
void Mul2() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float in2 = randomFloatFromBits();
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      ans[q] = in1[q] * in2;
    }
    Tensor3 a = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    Mul<1,size>(a,in2,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Mul2 : passed\n";
  }
  else {
    std::cout << "Mul2 : filed\n";
  }
}

template<int size>
void Div1() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float* in2 = (float*)malloc(sizeof(float) * size);
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      in2[q] = randomFloatFromBits();
      ans[q] = in1[q] / in2[q];
    }
    Tensor3 a = Create<1,size>();
    Tensor3 b = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    FromArray<1,size>(in2,b);
    Div<1,size>(a,b,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Div1 : passed\n";
  }
  else {
    std::cout << "Div1 : filed\n";
  }
}
template<int size>
void Div2() {
  bool pass = true;
  for(int q = 0;q < 10000;q++) {
    float* in1 = (float*)malloc(sizeof(float) * size);
    float in2 = randomFloatFromBits();
    float* out = (float*)malloc(sizeof(float) * size);
    float* ans = (float*)malloc(sizeof(float) * size);
    for(int q = 0;q < size;q++) {
      in1[q] = randomFloatFromBits();
      ans[q] = in1[q] / in2;
    }
    Tensor3 a = Create<1,size>();
    Tensor3 c = Create<1,size>();
    FromArray<1,size>(in1,a);
    Div<1,size>(a,in2,c);
    ToArray<1,size>(out, c);
    if(!check(size, out, ans)) {
      pass = false;
      break;
    }
  }
  if(pass) {
    std::cout << "Div2 : passed\n";
  }
  else {
    std::cout << "Div2 : filed\n";
  }
}

int main() {
    Plus1<512>();
    Plus2<512>();
    Sub1<512>();
    Sub2<512>();
    Mul1<512>();
    Mul2<512>();
    Div1<512>();
    Div2<512>();
    return 0;
}