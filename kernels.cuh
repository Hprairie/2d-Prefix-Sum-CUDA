#pragma once

#include <cub/cub.cuh>
#include "inputs.h"
#include "reverse_scan.cuh"

// ---------------------------------------
#define DIVUP(a, b) (a + b - 1)/b;

// constexpr dim3 BLOCK_SIZE {8, 1, 1};
// #define ROWS 1
// #define COLS 32
// #define ITEMS 4

constexpr dim3 BLOCK_SIZE {4, 2, 1};
#define ROWS 4
#define COLS 16
#define ITEMS 2
// --------------- Ops --------------------


struct ScanOp {
    __device__ __forceinline__ int operator()(const int &h, const int &t){
        return h + t;
    }
};

struct BlockPrefixCallbackOp
{
    // Running prefix
    int running_total;

    // Constructor
    __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}

    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int block_aggregate)
    {
        int old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

// -------------- Kernels -------------------
__global__ void block_scan(int *input, int *result); 

__global__ void reverse_block_scan(int *input, int* result); 


__global__ void warp_scan(int *input, int *result); 

__global__ void reverse_warp_scan(int *input, int *result);

__global__ void warp_scan_bidirectional(int *input, int *result);

__global__ void warp_scan2d(int *input, int * result);

__global__ void warp_scan2d_bidirectional(int *input, int *result);

// ---------------- Callers -------------------
