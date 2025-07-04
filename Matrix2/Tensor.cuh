#ifndef TENSOR
#define TENSOR

#define MAXTHREAD_PER_BLOCK 1024

class Tensor{
public:
    Tensor() {;}
    Tensor(Tensor& other) : data(other.data), pitch(other.pitch), row(other.row), col(other.col) {;}
    Tensor(const int row,const int col) : row(row), col(col) {
        cudaMallocPitch((void**)&data, &pitch, col * sizeof(float), row);
    }


    void toFloat(float* _data) noexcept {
        cudaMemcpy2D(_data, sizeof(float) * col, data, pitch, sizeof(float) * col, row, cudaMemcpyDeviceToHost);
    }
    
    float* data;
    std::size_t pitch;
    std::size_t row;
    std::size_t col;
};

__device__ float* getRow(const float* a, const int i, const std::size_t pitch) {
    return (float*)((char*)a + i * pitch);
}

__device__ float* get(const float* a, const int i, const int j, const std::size_t pitch) {
    return getRow(a,i,pitch) + j;
}

void fromArray(const float* in, Tensor& a) {
    cudaMemcpy2D(a.data, a.pitch, in, a.col * sizeof(float), a.col * sizeof(float), a.row, cudaMemcpyHostToDevice);
}

void toArray(const Tensor a, float* out) {
    cudaMemcpy2D(out, a.col * sizeof(float), a.data, a.pitch, a.col * sizeof(float), a.row, cudaMemcpyDeviceToHost);
}

__device__ int ceil(const int a, const int b) {
    return (a + b - 1) / b;
}

__global__ void plusKernel(const float* X, const float* Y,float* Z, const std::size_t pitch, const std::size_t N, const std::size_t M)  {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if(r < N && c < M) {
        const float* x = get(X,r,c,pitch);
        const float* y = get(Y,r,c,pitch);
        float* z = get(Z,r,c,pitch);
        z[0] = x[0] + y[0] * x[0] + 5.0f * x[0] * x[0] + y[0] * y[0] + 3.0f * y[0];
    }
}
void plusAsync(const Tensor a,const Tensor b,Tensor c) {
    dim3 blockDim(16,16);
    dim3 gridDim((a.row + 15) / 16, (a.col + 15) / 16);
    plusKernel<<<gridDim,blockDim>>>(a.data, b.data, c.data, a.pitch, a.row, a.col);
}

__global__ void PlusBatchKernel(
    const float* A, const float* B, const float* C,
    const std::size_t pitchA,const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t batch, const std::size_t row, const std::size_t col) {
    
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float b = *get(B, r, c, pitchB);
    if(r < row && c < col) {
        for(int i = 0;i < batch;i++) {
            *get(A, r + i * row, c, pitchA) = *get(C, r + i * row, c, pitchC) + b;
        }
    }
}
void PlusBatch(Tensor A, Tensor B, Tensor C,const int batch) {
    dim3 blockDim(16,16);
    dim3 gridDim((B.row + 15) / 16, (B.col + 15) / 16);
    PlusBatchKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, batch, B.row, B.col);
}


const int BLOCKSIZE = 16;

// A : d1 * d2
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelAB(
    const float* A, const float* B, float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    for(int i = 0;i < ceil(d2, BLOCKSIZE);i++) {
           
        const std::size_t loadIdxX = i * BLOCKSIZE + threadIdx.x;
        const std::size_t loadIdxY = i * BLOCKSIZE + threadIdx.y;

        if(r < d1 && loadIdxX < d2) {
            As[threadIdx.y][threadIdx.x] = get(A,r,loadIdxX,pitchA)[0];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(loadIdxY < d2 && c < d3) {
            Bs[threadIdx.y][threadIdx.x] = get(B,loadIdxY,c,pitchB)[0];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        if(r < d1 && c < d3) {
            for(int j = 0;j < BLOCKSIZE;j++) {
                CValue += As[threadIdx.y][j] * Bs[j][threadIdx.x];
            }
        }
        __syncthreads();
    }
    if(r < d1 && c < d3) {
        get(C,r,c,pitchC)[0] = CValue;
    }
}

// A : d2 * d1
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelATB(
    const float* A, const float* B, float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    for(int i = 0;i < ceil(d2, BLOCKSIZE);i++) {

        const std::size_t a0 = blockIdx.y * blockDim.y;
        const std::size_t b0 = blockIdx.x * blockDim.x;
        const std::size_t loadIdxY = i * BLOCKSIZE + threadIdx.y;

        if(loadIdxY < d2 && a0 + threadIdx.x < d1) {
            As[threadIdx.y][threadIdx.x] = *get(A,loadIdxY,a0 + threadIdx.x,pitchA);
            
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(loadIdxY < d2 && b0 + threadIdx.x < d3) {
            Bs[threadIdx.y][threadIdx.x] = *get(B,loadIdxY,b0 + threadIdx.x,pitchB);
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        if(r < d1 && c < d3) { 
            for(int j = 0;j < BLOCKSIZE;j++) {
                CValue += As[j][threadIdx.y] * Bs[j][threadIdx.x];
            }
        }
        __syncthreads();
    }
    if(r < d1 && c < d3) {
        *get(C,r,c,pitchC) = CValue;
    }
}


// A : d1 * d2
// B : d3 * d2
// C : d1 * d3
__global__ void MatMulKernelABT(
    const float* A, const float* B, const float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    if(r < d1 && c < d3) {
        for(int i = 0;i < ceil(d2, BLOCKSIZE);i++) {
            float a0 = blockIdx.y * blockDim.y;
            float b0 = blockIdx.x * blockDim.x;
            float loadIdxX = i * BLOCKSIZE + threadIdx.x;
            if(loadIdxX < d2) {
                As[threadIdx.y][threadIdx.x] = get(A,a0 + threadIdx.y,loadIdxX,pitchA)[0];
                Bs[threadIdx.y][threadIdx.x] = get(B,b0 + threadIdx.y,loadIdxX,pitchB)[0];
            }
            else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();
            for(int j = 0;j < BLOCKSIZE;j++) {
                CValue += As[threadIdx.y][j] * Bs[threadIdx.x][j];
            }
            __syncthreads();
        }
        get(C,r,c,pitchC)[0] = CValue;
    }
}



int ceilH(const int a, const int b) {
    return (a + b - 1) / b;
}

void MatMulPlusAsync(const Tensor a,const Tensor b, const Tensor c, bool ATransposed, bool BTransposed) {
    if(!ATransposed && !BTransposed) {
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
        dim3 gridDim(ceilH(c.col, BLOCKSIZE), ceilH(c.row, BLOCKSIZE));
        MatMulKernelAB<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.pitch, b.pitch, c.pitch, c.row, a.col, b.col);
    }
    else if(ATransposed && !BTransposed) {
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
        dim3 gridDim(ceilH(c.col, BLOCKSIZE), ceilH(c.row, BLOCKSIZE));
        MatMulKernelATB<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.pitch, b.pitch, c.pitch, c.row, a.row, c.col);
    }
    else if (!ATransposed && BTransposed) {
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
        dim3 gridDim(ceilH(c.col, BLOCKSIZE), ceilH(c.row, BLOCKSIZE));
        MatMulKernelABT<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.pitch, b.pitch, c.pitch, c.row, a.col, c.col);
    }
    else {
        // nothing implemented here.
    }
}

#endif