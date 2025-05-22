#ifndef TENSOR2
#define TENSOR2

#include "Header.h"


using Element = __m256;
using Tensor2 = const std::unique_ptr<Element, void(*)(void*)>;

consteval int getColSizeBytes(int col) {
    return (col / 8 + ((col & 7) > 0)) * 8 * 4;
}

consteval int getColSizeFloat(int col) {
    return (col / 8 + ((col & 7) > 0)) * 8;
}

consteval int getColSize(int col) {
    return (col / 8 + ((col & 7) > 0));
}

consteval int getSizeBytes(int d,int col) {
	return d * getColSizeBytes(col);
}

template<int d,int row,int col>
Tensor2 create() {
	consteval int realSize  = getSizeBytes<d,row,col>();
    void* data              = std::aligned_alloc(32, sizeof(Element) * realSize);
	return std::unique_ptr<Element, void(*)(void*)>(static_cast<Element*>(data), std::free);
}

// template<int d,int col>
// void fromFloat(float* f, Tensor2& a) {
//     char* itF               = reinterpret_cast<char*>(f);
//     char* data              = reinterpret_cast<char*>(a.get());
//     consteval int colSizeBytes  = getColSizeBytes<col>();
//     consteval int colDSizeBytes = col * sizeof(float);
//     for(int i = 0;i < d;i++) {
//         std::memcpy(data, itF, colDSizeBytes);
//         data += colSizeBytes;
//         itF += colDSizeBytes;
//     }
// }

// template<int d,int col>
// float* toFloat(Tensor2& a) {
//     float* f                = (float*)malloc(d * col * sizeof(float));
//     char* itF               = reinterpret_cast<char*>(f);
//     char* data              = reinterpret_cast<char*>(a.get());
//     consteval int colSizeBytes  = getColSizeBytes<col>();
//     consteval int colDSizeBytes = col * sizeof(float);
//     for(int i = 0;i < d;i++) {
//         std::memcpy(itF, data, colDSizeBytes);
//         data += colSizeBytes;
//         itF += colDSizeBytes;
//     }
//     return f;
// }

// template<int d,int col>
// void plus(const int d,const int col, Tensor2& a, Tensor2& b, Tensor2& c) {
//     consteval int loop = d * getColSize<d,col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_add_ps(a.get()[i], b.get()[i]);
// 	}
// }

// template<int d,int col>
// void plus(Tensor2& a, const float x, Tensor2& c) {
// 	const Element x_m256 = _mm256_set1_ps(x);
//     consteval int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_add_ps(a.get()[i], x_m256);
// 	}
// }

// template<int d,int col>
// void sub(Tensor2& a, Tensor2& b, Tensor2& c) {
// 	const int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_sub_ps(a.get()[i], b.get()[i]);
// 	}
// }

// template<int d,int col>
// void sub(Tensor2& a, const float x, Tensor2& c) {
// 	const Element x_m256 = _mm256_set1_ps(x);
//     consteval int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_sub_ps(a.get()[i], x_m256);
// 	}
// }

// template<int d,int col>
// void mul(Tensor2& a, Tensor2& b, Tensor2& c) {
// 	consteval int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_mul_ps(a.get()[i], b.get()[i]);
// 	}
// }

// template<int d,int col>
// void mul(Tensor2& a, const float x, Tensor2& c) {
// 	const Element x_m256 = _mm256_set1_ps(x);
//     consteval int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_mul_ps(a.get()[i], x_m256);
// 	}
// }

// template<int d,int col>
// void div(Tensor2& a, Tensor2& b, Tensor2& c) {
// 	consteval int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_div_ps(a.get()[i], b.get()[i]);
// 	}
// }

// template<int d,int col>
// void div(Tensor2& a, const float x, Tensor2& c) {
// 	const Element x_m256 = _mm256_set1_ps(x);
//     consteval int loop = d * getColSize<col>();
// 	for (int i = 0; i < loop; i++) {
// 		c.get()[i] = _mm256_div_ps(a.get()[i], x_m256);
// 	}
// }

// template<int d1,int d2,int d3>
// void MatMulAb(Tensor2 a, Tensor2 b, Tensor2& c) {
    
// }

// template<int d,int d1,int d2,int d3>
// void MatMulATbDirect(Tensor2& a,Tensor2& b,Tensor2& c) {
//     consteval int _d3 = getColSize<d3>();
//     consteval int _d1 = getColSize<d1>();
//     consteval int sizeA = _d1 * d2;
//     for(int i = 0;i < d2;i++) {
//         for(int j = 0;j < d2;j++) {
//             for(int kk = 0;kk < _d1;kk++) {
//                 for(int k = 0;k < 8;k++) {
//                     Element a = _mm256_set1_ps(((float*)&(a.get()[d * sizeA + i * _d1 + kk]))[k]);
//                     for(int w = 0;w < _d3;w++) {
//                         _mm256_storeu_ps(c.get()[(kk * 8 + k) * _d3 + w],_mm256_mul_ps(a, b.get()[j * _d3 + w]));
//                     }
//                 }
//             }
//         }
//     }
// }

// template<int d1,int d2,int d3>
// void MatMulATb(Tensor2 a, Tensor2 b, Tensor2& c) {
//     if constexpr(d3 >)
// }

// template<int d1,int d2,int d3>
// void MatMulATbT(Tensor2 a, Tensor2 b, Tensor2& c) {

// }

#endif