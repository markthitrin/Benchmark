#ifndef TENSOR3
#define TENSOR3

#include "Header.h"

using Tensor3 = float*;

constexpr int GetColSizeFloat(int col) {
    if(col % 8 == 0) {
        return col;
    }
    else {
        return (col / 8 * 8 + 1);
    }
}

constexpr int GetColSizeBytes(int col) {
    return GetColSizeFloat(col) * 4;
}

constexpr int GetSizeBytes(int d,int col) {
	return d * GetColSizeBytes(col);
}

template<int d,int col>
Tensor3 Create() {
	constexpr int realSize  = GetSizeBytes(d, col);
    void* data              = std::aligned_alloc(32, realSize);
	return (float*)data;
}

template<int d,int col>
Tensor3 Create0() {
	constexpr int realSize  = GetSizeBytes(d, col);
    void* data              = std::aligned_alloc(32, realSize);
    std::memset(data, 0, realSize);
	return (float*)data;
}

template<int d,int col>
void FromArray(float* f, Tensor3 A) {
    char* itF               = reinterpret_cast<char*>(f);
    char* data              = reinterpret_cast<char*>(A);
    constexpr int colSizeBytes  = GetColSizeBytes(col);
    constexpr int colDSizeBytes = col * sizeof(float);
    for(int i = 0;i < d;i++) {
        std::memcpy(data, itF, colDSizeBytes);
        data += colSizeBytes;
        itF += colDSizeBytes;
    }
}

template<int d,int col>
void ToArray(float* out, Tensor3 A) {
    char* itF               = reinterpret_cast<char*>(out);
    char* data              = reinterpret_cast<char*>(A);
    constexpr int colSizeBytes  = GetColSizeBytes(col);
    constexpr int colDSizeBytes = col * sizeof(float);
    for(int i = 0;i < d;i++) {
        std::memcpy(itF, data, colDSizeBytes);
        data += colSizeBytes;
        itF += colDSizeBytes;
    }
}

template<int d,int col>
void Plus(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] + b[i];
    }
}

template<int d,int col>
void Plus(Tensor3 A, const float x, Tensor3& C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] + x;
    }
}

template<int d,int col>
void Sub(Tensor3 A, Tensor3 B, Tensor3 C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] - b[i];
    }
}

template<int d,int col>
void Sub(Tensor3 A, const float x, Tensor3 C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] - x;
    }
}

template<int d,int col>
void Mul(Tensor3 A, Tensor3 B, Tensor3 C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] * b[i];
    }
}

template<int d,int col>
void Mul(Tensor3 A, const float x, Tensor3 C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] * x;
    }
}

template<int d,int col>
void Div(Tensor3 A, Tensor3 B, Tensor3 C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] / b[i];
    }
}

template<int d,int col>
void Div(Tensor3 A, const float x, Tensor3 C) {
	const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    const int _col = GetColSizeFloat(col);
    for(int i = 0;i < d * _col;i++) {
        c[i] = a[i] / x;
    }
}

template<int d,int col>
void Reset(Tensor3 A) {
    constexpr int realSize  = GetSizeBytes(d, col);
    void* data              = std::aligned_alloc(32, realSize);
    std::memset(data, 0, realSize);
}

template<int d1,int d2,int d3>
void MatMulPlus(Tensor3 A, Tensor3 B, Tensor3 C) {
    for(int i = 0;i < d1;i++) {
        for(int j = 0;j < d2;j++) {
            for(int k = 0;k < d3;k++) {
                C[i * d3 + j] += A[i * d2 + k] * B[k * d3 + j];
            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulPlus2(Tensor3 A, Tensor3 B, Tensor3 C) {
    for(int i = 0;i < d2;i++) {
        for(int j = 0;j < d1;j++) {
            for(int k = 0;k < d3;k++) {
                C[j * d3 + k] += A[j * d2 + i] * B[i * d2 + k];
            }
        }
    }
}


#endif