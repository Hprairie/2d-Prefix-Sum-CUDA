#include <cassert>
#include <cstddef>
#include <iostream>
#include "kernels.cuh"

void initialize(int* data) {
    for (int d=0; d < DIM; d++){
        for (int row=0; row<ROWS; row++) {
            for (int col=0; col<(COLS); col++) {
                data[(d * COLS*ROWS) + (row*COLS) + col] = col;
            }
        }
    }
}

void print_matrix(int* matrix) {
    for (int row=0; row<ROWS; row++) {
        printf("%3d [", row);
        for (int col=0; col<(COLS); col++) {
            const auto val = matrix[(row*COLS)+col];
            printf("(%3d) ", val);
        }

        printf("]\n");
    }
}

void check_matrix(int* matrix){
    int offset = ROWS * COLS;
    for (int row=0; row<ROWS; row++) {
        for (int col=0; col<(COLS); col++) {
            assert(matrix[(row*COLS)+col + offset] = matrix[(row*COLS)+col]);
        }
    }
}

void print_row(int* matrix){
    for (int row=0; row<1; row++) {
        printf("%3d [", row);
        for (int col=0; col<(COLS); col++) {
            const auto val = matrix[(row*COLS)+col];
            printf("(%3d) ", val);
        }

        printf("]\n");
    }
}

int main() {

    // --- Host Memory
    int h_matrix[DIM * ROWS * COLS] = {0};
    int h_result[DIM * ROWS * COLS] = {0};
    int h_result2[DIM * ROWS * COLS] = {0};

    // --- Initialize Data
    initialize(h_matrix);
    // print_matrix(h_matrix);
    // std::cout << "\n\n";

    // --- Device Memory
    int* d_matrix;
    int* d_result;
    int* d_result2;
    const size_t matrix_size = sizeof(int) * size_t(ROWS * COLS);

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_result, matrix_size);
    cudaMalloc((void**)&d_result2, matrix_size);
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // --- Kernel Launch
    dim3 block = BLOCK_SIZE;
    dim3 grid {DIM, 1, 1};

    warp_scan_orthoganal_2d<<<grid, block>>>(d_matrix, d_result, d_result2);
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
