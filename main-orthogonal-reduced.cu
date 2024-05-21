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
    int h_result2[ROWS * COLS] = {0};

    // --- Initialize Data
    initialize(h_matrix);
    // print_matrix(h_matrix);
    std::cout << "\n\n";

    // --- Device Memory
    int* d_matrix;
    int* d_result;
    int* d_result2;
    const size_t matrix_size = sizeof(int) * size_t(ROWS * COLS);

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_result, matrix_size);
    cudaMalloc((void**)&d_result2, matrix_size);
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // Mem calculations
    // typedef cub::WarpLoad<int, ITEMS, cub::WARP_LOAD_DIRECT, BLOCK_SIZE.x> WarpLoad;
    // typedef cub::WarpStore<int, ITEMS, cub::WARP_STORE_DIRECT, BLOCK_SIZE.x> WarpStore;
    // typedef cub::BlockExchange<int, BLOCK_SIZE.x, ITEMS, false, BLOCK_SIZE.y> BlockExchange;
    // typedef cub::WarpScan<int, BLOCK_SIZE.x> HorizontalWarpScan;
    // typedef WarpReverseScan<int, BLOCK_SIZE.x> HorizontalReverseWarpScan;
    // typedef cub::WarpScan<int, (int)(BLOCK_SIZE.y / ITEMS)> VerticalWarpScan;
    // typedef WarpReverseScan<int, (int)(BLOCK_SIZE.y / ITEMS)> VerticalReverseWarpScan;

    static constexpr int smem_size = std::max({
            sizeof(typename cub::WarpLoad<int, ITEMS, cub::WARP_LOAD_DIRECT, BLOCK_SIZE.x>::TempStorage) * BLOCK_SIZE.y,
            sizeof(typename cub::WarpStore<int, ITEMS, cub::WARP_STORE_DIRECT, BLOCK_SIZE.x>::TempStorage) * BLOCK_SIZE.y,
            sizeof(typename cub::BlockExchange<int, BLOCK_SIZE.x, ITEMS, false, BLOCK_SIZE.y>),
            sizeof(typename cub::WarpScan<int, BLOCK_SIZE.x>::TempStorage) * BLOCK_SIZE.y,
            sizeof(typename cub::WarpScan<int, (int)(BLOCK_SIZE.y / ITEMS)>::TempStorage) * BLOCK_SIZE.x * ITEMS
            });
    std::cout << smem_size << "\n";

    cudaStream_t stream = NULL;

    // --- Kernel Launch
    dim3 block = BLOCK_SIZE;
    dim3 grid {1, 1, 1};

    warp_scan_orthoganal_2d_shared<<<grid, block, smem_size, stream>>>(d_matrix, d_result, d_result2);
    cudaDeviceSynchronize();

    // --- Copy to Host
    cudaMemcpy(h_result, d_result, matrix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result2, d_result2, matrix_size, cudaMemcpyDeviceToHost);
    // std::cout << "\n";
    // print_matrix(h_result);
    // std::cout << "\n";
    // print_matrix(h_result2);

    return 0;
}
