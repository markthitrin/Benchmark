#ifndef TENSOR2
#define TENSOR2

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "Header.h"


using Element = __m256;
using Tensor2 = const std::unique_ptr<Element, void(*)(void*)>;

constexpr int getColSizeBytes(int col) {
    return (col / 8 + ((col & 7) > 0)) * 8 * 4;
}

constexpr int getColSizeFloat(int col) {
    return (col / 8 + ((col & 7) > 0)) * 8;
}

constexpr int getColSize(int col) {
    return (col / 8 + ((col & 7) > 0));
}

constexpr int getSizeBytes(int d,int col) {
	return d * getColSizeBytes(col);
}

template<int d,int col>
Tensor2 create() {
	constexpr int realSize  = getSizeBytes(d, col);
    void* data              = std::aligned_alloc(32, realSize);
	return std::unique_ptr<Element, void(*)(void*)>(
        static_cast<Element*>(data), std::free
    );
}

template<int d,int col>
Tensor2 create0() {
	constexpr int realSize  = getSizeBytes(d, col);
    void* data              = std::aligned_alloc(32, realSize);
    std::memset(data, 0, realSize);
	return std::unique_ptr<Element, void(*)(void*)>(
        static_cast<Element*>(data), std::free
    );
}

template<int d,int col>
void fromFloat(float* f, Tensor2& a) {
    char* itF               = reinterpret_cast<char*>(f);
    char* data              = reinterpret_cast<char*>(a.get());
    constexpr int colSizeBytes  = getColSizeBytes(col);
    constexpr int colDSizeBytes = col * sizeof(float);
    for(int i = 0;i < d;i++) {
        std::memcpy(data, itF, colDSizeBytes);
        data += colSizeBytes;
        itF += colDSizeBytes;
    }
}

template<int d,int col>
float* toFloat(Tensor2& a) {
    float* f                = (float*)malloc(d * col * sizeof(float));
    char* itF               = reinterpret_cast<char*>(f);
    char* data              = reinterpret_cast<char*>(a.get());
    constexpr int colSizeBytes  = getColSizeBytes(col);
    constexpr int colDSizeBytes = col * sizeof(float);
    for(int i = 0;i < d;i++) {
        std::memcpy(itF, data, colDSizeBytes);
        data += colSizeBytes;
        itF += colDSizeBytes;
    }
    return f;
}

template<int d,int col>
void plus(Tensor2& a, Tensor2& b, Tensor2& c) {
    constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_add_ps(a.get()[i], b.get()[i]);
	}
}

template<int d,int col>
void plus(Tensor2& a, const float x, Tensor2& c) {
	const Element x_m256 = _mm256_set1_ps(x);
    constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_add_ps(a.get()[i], x_m256);
	}
}

template<int d,int col>
void sub(Tensor2& a, Tensor2& b, Tensor2& c) {
	const int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_sub_ps(a.get()[i], b.get()[i]);
	}
}

template<int d,int col>
void sub(Tensor2& a, const float x, Tensor2& c) {
	const Element x_m256 = _mm256_set1_ps(x);
    constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_sub_ps(a.get()[i], x_m256);
	}
}

template<int d,int col>
void mul(Tensor2& a, Tensor2& b, Tensor2& c) {
	constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_mul_ps(a.get()[i], b.get()[i]);
	}
}

template<int d,int col>
void mul(Tensor2& a, const float x, Tensor2& c) {
	const Element x_m256 = _mm256_set1_ps(x);
    constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_mul_ps(a.get()[i], x_m256);
	}
}

template<int d,int col>
void div(Tensor2& a, Tensor2& b, Tensor2& c) {
	constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_div_ps(a.get()[i], b.get()[i]);
	}
}

template<int d,int col>
void div(Tensor2& a, const float x, Tensor2& c) {
	const Element x_m256 = _mm256_set1_ps(x);
    constexpr int loop = d * getColSize(col);
	for (int i = 0; i < loop; i++) {
		c.get()[i] = _mm256_div_ps(a.get()[i], x_m256);
	}
}

// template<int d,int d1,int d2,int d3>
// void MatMulAb1(Tensor2 a, Tensor2 b, Tensor2& c) {
//     constexpr int _d3 = getColSize(d3);
//     for(int i 0;i < _d3;i++) {
//         c.get()[i] = _mm256_setzero_ps();
//     }


// }

// template<int d1,int d2,int d3>
// void MatMulaTbDirect(const Element* const a,const Element* const b,Element* const c) {
//     constexpr int _d3 = getColSize(d3);
//     constexpr int _d1 = getColSize(d1);
//     constexpr int sizeA = _d1 * d2;
//     for(int i = 0;i < d2;i++) {
//         for(int j = 0;j < d2;j++) {
//             for(int k = 0;k < _d1 * 8;k++) {
//                 Element a = _mm256_set1_ps(((float*)&(a[i * _d1 + k / 8]))[k & 7]);
//                 for(int w = 0;w < _d3;w++) {
//                     _mm256_storeu_ps(c[k * _d3 + w],_mm256_mul_ps(a, b[j * _d3 + w]));
//                 }
//             }
//         }
//     }
// }

// N is always 8 and M is at max 112 ans must divisible by 8
// Therefore _M is int and max at 14
template<int _M,int _d3>
void MatMulaTbBlock(const Element* a,const Element* b,Element* const c) {
    for(int i = 0;i < 8;i++) {
        Element ai = _mm256_set1_ps(((float*)(a))[i]);
        for(int j = 0;j < _M;j++) {
            c[i * _d3 + j] = _mm256_add_ps(_mm256_mul_ps(ai, b[j]),c[i * _d3 + j]);
        }
    }
}

template<int d1,int d2,int d3>
void MatMulaTb(Tensor2& a, Tensor2& b, Tensor2& c) {
    constexpr int _d3 = getColSize(d3);
    constexpr int _d1 = getColSize(d1);
    constexpr int _M = std::min(14, _d3);
    std::memset(c.get(), 0, _d1 * 8 * _d3 * 8 * sizeof(float));
    const Element* ita = a.get();
    const Element* itb = b.get();
    Element* itc = c.get(); 
    for(int i = 0;i < d2;i++) {
        for(int j = 0;j < _d1;j++) {
            int k = 0;
            for(;k < (_d3 / _M) * _M;k+=_M) {
                MatMulaTbBlock<_M,_d3>(ita,itb,itc);
                itb += _M;
                itc += _M;
            }
            constexpr int _remainM = _d3 - (_d3 / _M) * _M;
            MatMulaTbBlock<_remainM,_d3>(ita,itb,itc);
            itb += _remainM - _d3;
            itc += _remainM + (7 * _d3);
            ita ++;
        }
        itb += _d3;
        itc = c.get();
    }
}

// template<int d,int d1,int d2,int d3>
// void MatMulATbDirect(Tensor2& a,Tensor2& b,Tensor2& c) {
//     constexpr int _d3 = getColSize(d3);
//     constexpr int _d1 = getColSize(d1);
//     constexpr int sizeA = _d1 * d2;
//     for(int i = 0;i < d2;i++) {
//         for(int j = 0;j < d2;j++) {
//             for(int k = 0;k < _d1 * 8;k++) {
//                 Element a = _mm256_set1_ps(((float*)&(a.get()[d * sizeA + i * _d1 + kk]))[k & 7]);
//                 for(int w = 0;w < _d3;w++) {
//                     _mm256_storeu_ps(c.get()[(kk * 8 + k) * _d3 + w],_mm256_mul_ps(a, b.get()[j * _d3 + w]));
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


#pragma GCC diagnostic pop

#endif