#ifndef TENSOR3
#define TENSOR3

#include "Header.h"

using Tensor3 = float*;

constexpr int GetColSizeFloat(int col) {
    if(col % 8 == 0) {
        return col;
    }
    else {
        return ((col / 8 + 1) * 8);
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
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int _d2 = GetColSizeFloat(d2);
    for(int i = 0;i < d1;i++) {
        for(int j = 0;j < _d3;j++) {
            for(int k = 0;k < _d2;k++) {
                c[i * _d3 + j] += a[i * _d2 + k] * b[k * _d3 + j];
            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulPlus2(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int _d2 = GetColSizeFloat(d2);
    for(int i = 0;i < _d2;i++) {
        for(int j = 0;j < d1;j++) {
            for(int k = 0;k < _d3;k++) {
                c[j * _d3 + k] += a[j * _d2 + i] * b[i * _d3 + k];
            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulPlus3(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int _d2 = GetColSizeFloat(d2);
    constexpr int BLOCK_SIZE = 16;
    for (int jj = 0; jj < _d3; jj += BLOCK_SIZE) {
       const int jjTempMin = std::min(jj + BLOCK_SIZE,_d3); 
       for (int kk = 0; kk < _d2; kk += BLOCK_SIZE) {
           const int kkTempMin = std::min(kk + BLOCK_SIZE,_d2); 
           for (int i = 0; i < d1; i++) {
               for (int k = kk ; k < kkTempMin ; k++) {
                   for (int j = jj; j < jjTempMin; j++) {
                      c[i * _d3 + j]  +=  a[i * _d2 + k] * b[k * _d3 + j];
                   }
               }
           }
      }
   }
}

template<int d1,int d2,int d3>
void MatMulPlus4(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int _d2 = GetColSizeFloat(d2);
    constexpr int BLOCK_SIZE = 64;

    constexpr int _ii = d1 / BLOCK_SIZE * BLOCK_SIZE;
    constexpr int _jj = _d3 / BLOCK_SIZE * BLOCK_SIZE;
    for(int ii = 0;ii < _ii;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE) {
            for(int k = 0;k < _d2;k++) {
                for(int i = 0;i < BLOCK_SIZE;i++) {
                    for(int j = 0;j < BLOCK_SIZE;j++) {
                        c[(ii + i) * _d3 + (jj + j)]  +=  a[(ii + i) * _d2 + k] * b[k * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < _d2;k++) {
            for(int i = 0;i < BLOCK_SIZE;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE;j++) {
                    c[(ii + i) * _d3 + (_jj + j)]  +=  a[(ii + i) * _d2 + k] * b[k * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE) {
        for(int k = 0;k < _d2;k++) {
            for(int i = 0;i < d1 % BLOCK_SIZE;i++) {
                for(int j = 0;j < BLOCK_SIZE;j++) {
                    c[(_ii + i) * _d3 + (jj + j)]  +=  a[(_ii + i) * _d2 + k] * b[k * _d3 + (jj + j)];
                }
            } 
        }
    }
    for(int k = 0;k < _d2;k++) {
        for(int i = 0;i < d1 % BLOCK_SIZE;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE;j++) {
                c[(_ii + i) * _d3 + (_jj + j)]  +=  a[(_ii + i) * _d2 + k] * b[k * _d3 + (_jj + j)];
            }
        } 
    }
}

template<int d1,int d2,int d3>
void MatMulPlus5(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d1 = GetColSizeFloat(d1);
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int BLOCK_SIZE = 64;

    constexpr int _ii = _d1 / BLOCK_SIZE * BLOCK_SIZE;
    constexpr int _jj = _d3 / BLOCK_SIZE * BLOCK_SIZE;
    for(int ii = 0;ii < _ii;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE) {
            for(int k = 0;k < d2;k++) {
                for(int i = 0;i < BLOCK_SIZE;i++) {
                    for(int j = 0;j < BLOCK_SIZE;j++) {
                        c[(ii + i) * _d3 + (jj + j)]  +=  a[k * _d1 + (ii + i)] * b[k * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < d2;k++) {
            for(int i = 0;i < BLOCK_SIZE;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE;j++) {
                    c[(ii + i) * _d3 + (_jj + j)]  +=  a[k * _d1 + (ii + i)] * b[k * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE) {
        for(int k = 0;k < d2;k++) {
            for(int i = 0;i < _d1 % BLOCK_SIZE;i++) {
                for(int j = 0;j < BLOCK_SIZE;j++) {
                    c[(_ii + i) * _d3 + (jj + j)]  +=  a[k * _d1 + (_ii + i)] * b[k * _d3 + (jj + j)];
                }
            } 
        }
    }
    for(int k = 0;k < d2;k++) {
        for(int i = 0;i < _d1 % BLOCK_SIZE;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE;j++) {
                c[(_ii + i) * _d3 + (_jj + j)]  +=  a[k * _d1 + (_ii + i)] * b[k * _d3 + (_jj + j)];
            }
        } 
    }
}

template<int d1,int d2,int d3>
void MatMulPlus6(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d1 = GetColSizeFloat(d1);
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int BLOCK_SIZE = 80;

    constexpr int _ii = _d1 / BLOCK_SIZE * BLOCK_SIZE;
    constexpr int _jj = _d3 / BLOCK_SIZE * BLOCK_SIZE;
    for(int ii = 0;ii < _ii;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE) {
            for(int k = 0;k < d2;k++) {
                for(int i = 0;i < BLOCK_SIZE;i++) {
                    for(int j = 0;j < BLOCK_SIZE;j++) {
                        c[(ii + i) * _d3 + (jj + j)]  +=  a[k * _d1 + (ii + i)] * b[k * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < d2;k++) {
            for(int i = 0;i < BLOCK_SIZE;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE;j++) {
                    c[(ii + i) * _d3 + (_jj + j)]  +=  a[k * _d1 + (ii + i)] * b[k * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE) {
        for(int k = 0;k < d2;k++) {
            for(int i = 0;i < _d1 % BLOCK_SIZE;i++) {
                for(int j = 0;j < BLOCK_SIZE;j++) {
                    c[(_ii + i) * _d3 + (jj + j)]  +=  a[k * _d1 + (_ii + i)] * b[k * _d3 + (jj + j)];
                }
            } 
        }
    }
    for(int k = 0;k < d2;k++) {
        for(int i = 0;i < _d1 % BLOCK_SIZE;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE;j++) {
                c[(_ii + i) * _d3 + (_jj + j)]  +=  a[k * _d1 + (_ii + i)] * b[k * _d3 + (_jj + j)];
            }
        } 
    }
}

template<int d1,int d2,int d3>
void MatMulPlus7(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d1 = GetColSizeFloat(d1);
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int l = 32 * 1024 / (_d3 * 4);
    constexpr int BLOCK_SIZE1 = 8;
    constexpr int BLOCK_SIZE3 = 512;

    constexpr int _ii = _d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
    constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
    for(int ii = 0;ii < _ii;ii += BLOCK_SIZE1) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
            for(int k = 0;k < d2;k++) {
                for(int i = 0;i < BLOCK_SIZE1;i++) {
                    for(int j = 0;j < BLOCK_SIZE3;j++) {
                        c[(ii + i) * _d3 + (jj + j)]  +=  a[k * _d1 + (ii + i)] * b[k * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < d2;k++) {
            for(int i = 0;i < BLOCK_SIZE1;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                    c[(ii + i) * _d3 + (_jj + j)]  +=  a[k * _d1 + (ii + i)] * b[k * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
        for(int k = 0;k < d2;k++) {
            for(int i = 0;i < _d1 % BLOCK_SIZE1;i++) {
                for(int j = 0;j < BLOCK_SIZE3;j++) {
                    c[(_ii + i) * _d3 + (jj + j)]  +=  a[k * _d1 + (_ii + i)] * b[k * _d3 + (jj + j)];
                }
            } 
        }
    }
    for(int k = 0;k < d2;k++) {
        for(int i = 0;i < _d1 % BLOCK_SIZE1;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                c[(_ii + i) * _d3 + (_jj + j)]  +=  a[k * _d1 + (_ii + i)] * b[k * _d3 + (_jj + j)];
            }
        } 
    }
}

template<int d1,int d2,int d3>
void MatMulPlus8(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d2 = GetColSizeFloat(d2);
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int l = 32 * 1024 / (_d3 * 4);
    constexpr int BLOCK_SIZE1 = 8;
    constexpr int BLOCK_SIZE3 = 512;

    constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
    constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
    for(int ii = 0;ii < _ii;ii += BLOCK_SIZE1) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
            for(int k = 0;k < _d2;k++) {
                for(int i = 0;i < BLOCK_SIZE1;i++) {
                    for(int j = 0;j < BLOCK_SIZE3;j++) {
                        c[(ii + i) * _d3 + (jj + j)]  +=  a[(ii + i) * _d2 + k] * b[k * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < _d2;k++) {
            for(int i = 0;i < BLOCK_SIZE1;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                    c[(ii + i) * _d3 + (_jj + j)]  +=  a[(ii + i) * _d2 + k] * b[k * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
        for(int k = 0;k < _d2;k++) {
            for(int i = 0;i < d1 % BLOCK_SIZE1;i++) {
                for(int j = 0;j < BLOCK_SIZE3;j++) {
                    c[(_ii + i) * _d3 + (jj + j)]  +=  a[(_ii + i) * _d2 + k] * b[k * _d3 + (jj + j)];
                }
            } 
        }
    }
    for(int k = 0;k < _d2;k++) {
        for(int i = 0;i < d1 % BLOCK_SIZE1;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                c[(_ii + i) * _d3 + (_jj + j)]  +=  a[(_ii + i) * _d2 + k] * b[k * _d3 + (_jj + j)];
            }
        } 
    }
}

template<int d1,int d2,int d3>
void MatMulPlus9(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d1 = GetColSizeFloat(d1);
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int BLOCK_SIZE1 =
        (d2 <= 8) ? _d1 :
        (_d3 > 256) ? 8 :
        16;

    constexpr int BLOCK_SIZE2 = 
        (_d3 > 256) ? 16 :
        (_d3 > 128) ? 32 :
        16 * 1024 / (64 + _d3 * 4);

    constexpr int BLOCK_SIZE3 =
        (d2 <= 8) ? _d3 : 
        (_d3 > 512) ? 384 :
        (_d3 > 256) ? 256 :
        (_d3 > 128) ? 256 :
        128;

    constexpr int _ii = _d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
    constexpr int _kk = _kk / BLOCK_SIZE2 * BLOCK_SIZE2;
    constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
    for(int kk = 0;kk < _kk;kk++) {
        for(int ii = 0;ii < _ii;ii += BLOCK_SIZE1) {
            for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
                for(int k = 0;k < BLOCK_SIZE2;k++) {
                    for(int i = 0;i < BLOCK_SIZE1;i++) {
                        for(int j = 0;j < BLOCK_SIZE3;j++) {
                            c[(ii + i) * _d3 + (jj + j)]  +=  a[(kk + k) * _d1 + (ii + i)] * b[(kk + k) * _d3 + (jj + j)];
                        }
                    } 
                }
            }
            for(int k = 0;k < BLOCK_SIZE2;k++) {
                for(int i = 0;i < BLOCK_SIZE1;i++) {
                    for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                        c[(ii + i) * _d3 + (_jj + j)]  +=  a[(kk + k) * _d1 + (ii + i)] * b[(kk + k) * _d3 + (_jj + j)];
                    }
                } 
            }
        }
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
            for(int k = 0;k < BLOCK_SIZE2;k++) {
                for(int i = 0;i < _d1 % BLOCK_SIZE1;i++) {
                    for(int j = 0;j < BLOCK_SIZE3;j++) {
                        c[(_ii + i) * _d3 + (jj + j)]  +=  a[(kk + k) * _d1 + (_ii + i)] * b[(kk + k) * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < BLOCK_SIZE2;k++) {
            for(int i = 0;i < _d1 % BLOCK_SIZE1;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                    c[(_ii + i) * _d3 + (_jj + j)]  +=  a[(kk + k) * _d1 + (_ii + i)] * b[(kk + k) * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int ii = 0;ii < _ii;ii += BLOCK_SIZE1) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
            for(int k = 0;k < d2 % BLOCK_SIZE2;k++) {
                for(int i = 0;i < BLOCK_SIZE1;i++) {
                    for(int j = 0;j < BLOCK_SIZE3;j++) {
                        c[(ii + i) * _d3 + (jj + j)]  +=  a[(_kk + k) * _d1 + (ii + i)] * b[(_kk + k) * _d3 + (jj + j)];
                    }
                } 
            }
        }
        for(int k = 0;k < d2 % BLOCK_SIZE2;k++) {
            for(int i = 0;i < BLOCK_SIZE1;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                    c[(ii + i) * _d3 + (_jj + j)]  +=  a[(_kk + k) * _d1 + (ii + i)] * b[(_kk + k) * _d3 + (_jj + j)];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
        for(int k = 0;k < d2 % BLOCK_SIZE2;k++) {
            for(int i = 0;i < _d1 % BLOCK_SIZE1;i++) {
                for(int j = 0;j < BLOCK_SIZE3;j++) {
                    c[(_ii + i) * _d3 + (jj + j)]  +=  a[(_kk + k) * _d1 + (_ii + i)] * b[(_kk + k) * _d3 + (jj + j)];
                }
            } 
        }
    }
    for(int k = 0;k < d2 % BLOCK_SIZE2;k++) {
        for(int i = 0;i < _d1 % BLOCK_SIZE1;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                c[(_ii + i) * _d3 + (_jj + j)]  +=  a[(_kk + k) * _d1 + (_ii + i)] * b[(_kk + k) * _d3 + (_jj + j)];
            }
        } 
    }
}

template<int d1,int d2,int d3>
void MatMulPlus10(Tensor3 A, Tensor3 B, Tensor3 C) {
    const float* a = static_cast<const float*>(__builtin_assume_aligned(A, 32));
    const float* b = static_cast<const float*>(__builtin_assume_aligned(B, 32));
    float* c = static_cast<float*>(__builtin_assume_aligned(C, 32));
    constexpr int _d2 = GetColSizeFloat(d2);
    constexpr int _d3 = GetColSizeFloat(d3);
    constexpr int BLOCK_SIZE1 =
        32;

    constexpr int BLOCK_SIZE2 = 
        32;
    
    constexpr int BLOCK_SIZE3 =
        32;

    constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
    constexpr int _kk = _d2 / BLOCK_SIZE2 * BLOCK_SIZE2;
    constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
    for(int ii = 0;ii < _ii;ii+=BLOCK_SIZE1) {
        for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
            for(int kk = 0;kk < _kk;kk += BLOCK_SIZE2) {
                for(int i = 0;i < BLOCK_SIZE1;i++) {
                    for(int j = 0;j < BLOCK_SIZE3;j++) {
                        for(int k = 0;k < BLOCK_SIZE2;k++) {
                            c[(ii + i) * _d3 + jj + j] += a[(ii + i) * _d2 + kk + k] * b[(kk + k) * _d3 + jj + j];
                        }
                    } 
                }
            }
            for(int i = 0;i < BLOCK_SIZE1;i++) {
                for(int j = 0;j < BLOCK_SIZE3;j++) {
                    for(int k = 0;k < _d2 % BLOCK_SIZE2;k++) {
                        c[(ii + i) * _d3 + jj + j] += a[(ii + i) * _d2 + _kk + k] * b[(_kk + k) * _d3 + jj + j];
                    }
                } 
            }
        }
        for(int kk = 0;kk < _kk;kk += BLOCK_SIZE2) {
            for(int i = 0;i < BLOCK_SIZE1;i++) {
                for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                    for(int k = 0;k < BLOCK_SIZE2;k++) {
                        c[(ii + i) * _d3 + _jj + j] += a[(ii + i) * _d2 + kk + k] * b[(kk + k) * _d3 + _jj + j];
                    }
                } 
            }
        }
        for(int i = 0;i < BLOCK_SIZE1;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                for(int k = 0;k < _d2 % BLOCK_SIZE2;k++) {
                    c[(ii + i) * _d3 + _jj + j] += a[(ii + i) * _d2 + _kk + k] * b[(_kk + k) * _d3 + _jj + j];
                }
            } 
        }
    }
    for(int jj = 0;jj < _jj;jj += BLOCK_SIZE3) {
        for(int kk = 0;kk < _kk;kk += BLOCK_SIZE2) {
            for(int i = 0;i < d1 % BLOCK_SIZE1;i++) {
                for(int j = 0;j < BLOCK_SIZE3;j++) {
                    for(int k = 0;k < BLOCK_SIZE2;k++) {
                        c[(_ii + i) * _d3 + jj + j] += a[(_ii + i) * _d2 + kk + k] * b[(kk + k) * _d3 + jj + j];
                    }
                } 
            }
        }
        for(int i = 0;i < d1 % BLOCK_SIZE1;i++) {
            for(int j = 0;j < BLOCK_SIZE3;j++) {
                for(int k = 0;k < _d2 % BLOCK_SIZE2;k++) {
                    c[(_ii + i) * _d3 + jj + j] += a[(_ii + i) * _d2 + _kk + k] * b[(_kk + k) * _d3 + jj + j];
                }
            } 
        }
    }
    for(int kk = 0;kk < _kk;kk += BLOCK_SIZE2) {
        for(int i = 0;i < d1 % BLOCK_SIZE1;i++) {
            for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
                for(int k = 0;k < BLOCK_SIZE2;k++) {
                    c[(_ii + i) * _d3 + _jj + j] += a[(_ii + i) * _d2 + kk + k] * b[(kk + k) * _d3 + _jj + j];
                }
            } 
        }
    }
    for(int i = 0;i < d1 % BLOCK_SIZE1;i++) {
        for(int j = 0;j < _d3 % BLOCK_SIZE3;j++) {
            for(int k = 0;k < _d2 % BLOCK_SIZE2;k++) {
                c[(_ii + i) * _d3 + _jj + j] += a[(_ii + i) * _d2 + _kk + k] * b[(_kk + k) * _d3 + _jj + j];
            }
        } 
    }
}


#endif