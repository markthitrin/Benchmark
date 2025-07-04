
template<int row,int col>
class Matrix {
public:
    Matrix() {
        data = new float[row * col];
    }
    ~Matrix() {
        delete[] data;
    }
    float* data;
};
constexpr int BLOCK_SIZE = 96;

template<int d1,int d2,int d3>
void MatMulABPlus(Matrix<d1, d2>& A, Matrix<d2, d3>& B, Matrix<d1, d3>& C) {
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;i < BLOCK_SIZE && ii + i < d1;i++) {
                    for(int k = 0; k < BLOCK_SIZE && kk + k < d2;k++) {
                        float a = A.data[(ii + i) * d2 + (kk + k)];
                        for(int j = 0;j < BLOCK_SIZE && jj + j < d3;j++) {
                            C.data[(ii + i) * d3 + (jj + j)] += a * B.data[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulATBPlus(Matrix<d2, d1>& A, Matrix<d2, d3>& B, Matrix<d1, d3>& C) {
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;i < BLOCK_SIZE && ii + i < d1;i++) {
                    for(int k = 0; k < BLOCK_SIZE && kk + k < d2;k++) {
                        float a = A.data[(kk + k) * d1 + (ii + i)];
                        for(int j = 0;j < BLOCK_SIZE && jj + j < d3;j++) {
                            C.data[(ii + i) * d3 + (jj + j)] += a * B.data[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulABTPlus(Matrix<d1, d2>& A, Matrix<d3, d2>& B, Matrix<d1, d3>& C) {
    static Matrix<d2, d3> BT; // parallel hazard
    for(int ii = 0;ii < d2;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int i = 0;i < BLOCK_SIZE && ii + i < d2;i++){
                for(int j = 0;j < BLOCK_SIZE && jj + j < d3;j++) {
                    BT.data[(ii + i) * d3 + jj + j] = B.data[(jj + j) * d2 + ii + i];
                }
            }
        }
    }
    MatMulABPlus(A, BT, C);
}