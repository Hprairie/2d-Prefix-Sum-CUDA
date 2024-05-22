#include <cassert>
#include <cstddef>
#include <iostream>
#include "kernels.cuh"

__global__ void print_kernel(float *input, float* result, float* result2){
    if (blockIdx.x == 0){
        if (threadIdx.y == 0){
            printf("%f ", input[threadIdx.x]);
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initialize(float* data) {
    for (int d=0; d < DIM; d++){
        for (int row=0; row<ROWS; row++) {
            for (int col=0; col<(COLS); col++) {
                data[(d * COLS*ROWS) + (row*COLS) + col] = (float)col;
            }
        }
    }
}

void print_matrix(float* matrix) {
    for (int row=0; row<ROWS; row++) {
        printf("%3d [", row);
        for (int col=0; col<(COLS); col++) {
            const auto val = matrix[(row*COLS)+col];
            printf("(%.0f) ", val);
        }

        printf("]\n");
    }
}

void check_matrix(float* matrix){
    int offset = ROWS * COLS;
    for (int row=0; row<ROWS; row++) {
        for (int col=0; col<(COLS); col++) {
            assert(matrix[(row*COLS)+col + offset] = matrix[(row*COLS)+col]);
        }
    }
}

void print_row(float* matrix){
    for (int row=0; row<1; row++) {
        printf("%3d [", row);
        for (int col=0; col<(COLS); col++) {
            const auto val = matrix[(row*COLS)+col];
            printf("(%.0f) ", val);
        }
        printf("]\n");
    }
}

int main() {

    // --- Host Memory
    float h_matrix[DIM * ROWS * COLS] = {0};
    float h_result[DIM * ROWS * COLS] = {0};
    float h_result2[DIM * ROWS * COLS] = {0};

    // --- Initialize Data
    initialize(h_matrix);
    print_row(h_matrix);
    // print_matrix(h_matrix);
    // std::cout << "\n\n";

    // --- Device Memory
    float* d_matrix;
    float* d_result;
    float* d_result2;
    const size_t matrix_size = sizeof(float) * size_t(DIM * ROWS * COLS);

    gpuErrchk(cudaMalloc((void**)&d_matrix, matrix_size));
    gpuErrchk(cudaMalloc((void**)&d_result, matrix_size));
    gpuErrchk(cudaMalloc((void**)&d_result2, matrix_size));
    gpuErrchk(cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice));

    // --- Kernel Launch
    dim3 block = BLOCK_SIZE;
    dim3 grid {DIM, 1, 1};

    print_kernel<<<grid, block>>>(d_matrix, d_result, d_result2);
    cudaGetLastError();
    cudaDeviceSynchronize();

    // CubDebugExit(cudaDeviceSynchronize());

    // --- Copy to Host
    cudaMemcpy(h_result, d_result, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result2, d_result2, matrix_size, cudaMemcpyDeviceToHost);
    print_row(h_result);
    print_row(h_result2);
    // std::cout << "\n";
    // print_matrix(h_result);
    // std::cout << "\n";
    // print_matrix(h_result2);

    return 0;
}
