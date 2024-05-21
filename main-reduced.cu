#include <algorithm>
#include <cstddef>
#include <iostream>
#include "kernels.cuh"

void initialize(int* data) {
    for (int row=0; row<ROWS; row++) {
        for (int col=0; col<(COLS); col++) {
            data[(row*COLS) + col] = col;
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

void format_matrix(int* matrix){

}


int main() {

    // --- Host Memory
    int h_matrix[ROWS * COLS] = {0};
    int h_result[ROWS * COLS] = {0};

    // --- Initialize Data
    initialize(h_matrix);
    //print_matrix(h_matrix);
    std::cout << "\n\n";

    // --- Device Memory
    int* d_matrix;
    int* d_result;
    const size_t matrix_size = sizeof(int) * size_t(ROWS * COLS);

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_result, matrix_size);
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    //generate smem total
    static constexpr int smem_size = std::max({
            sizeof(typename cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y>::TempStorage),
            sizeof(typename cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT, BLOCK_SIZE.y>::TempStorage),
            sizeof(typename cub::WarpScan<int, BLOCK_SIZE.x>::TempStorage) * BLOCK_SIZE.y
            });
    std::cout << smem_size << "\n";

    cudaStream_t stream = NULL;

    // --- Kernel Launch
    dim3 block = BLOCK_SIZE;
    dim3 grid {1, 1, 1};

    shared_warp_scan<<<grid, block, smem_size, stream>>>(d_matrix, d_result);
    cudaDeviceSynchronize();

    // --- Copy to Host
    cudaMemcpy(h_result, d_result, matrix_size, cudaMemcpyDeviceToHost);
    // std::cout << "\n\n";
    // print_matrix(h_result);

    return 0;
}
