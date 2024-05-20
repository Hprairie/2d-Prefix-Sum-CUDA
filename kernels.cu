
#include "kernels.cuh"
#include "reverse_scan.cuh"

// -------------- Block Scan Kernels -------------------
__global__ void block_scan(int *input, int *result) {
    typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT, BLOCK_SIZE.y> BlockStore;
    typedef cub::BlockScan<int, BLOCK_SIZE.x, cub::BLOCK_SCAN_RAKING, BLOCK_SIZE.y> BlockScan;

    auto valid_items = BLOCK_SIZE.x * BLOCK_SIZE.y;
    if (threadIdx.x == 0 && threadIdx.y == 0){
        printf("Value of data %d \n", input[1]);
    }

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockScan::TempStorage scan;
        typename BlockStore::TempStorage store;
    } temp_storage;


    int thread_data[ITEMS];
    BlockLoad(temp_storage.load).Load(input, thread_data, valid_items);
    __syncthreads();

    
    printf("Thread %d,%d: %d \n", threadIdx.x, threadIdx.y, thread_data[0]);
    BlockScan(temp_storage.scan).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    if (threadIdx.x == 4 && threadIdx.y == 0){
        printf("VAlue of data end %d \n", thread_data[0]);
    }

    BlockStore(temp_storage.store).Store(result, thread_data, valid_items);
    __syncthreads();
}

__global__ void reverse_block_scan(int *input, int* result){
    typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT> BlockStore;
    typedef BlockReverseScan<int, BLOCK_SIZE.x, false> BlockScan;


    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockScan::TempStorage scan;
        typename BlockStore::TempStorage store;
    } temp_storage;


    int thread_data[ITEMS];
    BlockLoad(temp_storage.load).Load(input, thread_data);
    __syncthreads();

    
    printf("Thread %d,%d: %d \n", threadIdx.x, threadIdx.y, thread_data[0]);
    auto scan_op = ScanOp();
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(20);

    BlockScan(temp_storage.scan).ExclusiveReverseScan(thread_data, thread_data, scan_op, prefix_op);
    __syncthreads();

    if (threadIdx.x == 4 && threadIdx.y == 0){
        printf("VAlue of data end %d \n", thread_data[0]);
    }

    BlockStore(temp_storage.store).Store(result, thread_data);
    __syncthreads();

}

// -------------- Warp Scan Kernels -------------------

__global__ void warp_scan(int *input, int *result){

    typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT, BLOCK_SIZE.y> BlockStore;
    typedef cub::WarpScan<int, BLOCK_SIZE.x> WarpScan;

    if (threadIdx.x == 0 && threadIdx.y == 0){
        printf("Value of data %d \n", input[1]);
    }

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename WarpScan::TempStorage scan[BLOCK_SIZE.y];
        typename BlockStore::TempStorage store;
    } temp_storage;


    constexpr int scans = ROWS / BLOCK_SIZE.y;
    for (int x = 0; x < scans; x ++){
        int thread_data[ITEMS];
        BlockLoad(temp_storage.load).Load(input + x * blockDim.y * COLS, thread_data);

        auto scan_op = cub::Sum();
        int prefix = cub::internal::ThreadReduce(thread_data, scan_op);

        int warp_id = threadIdx.x / BLOCK_SIZE.x;
        WarpScan(temp_storage.scan[warp_id]).ExclusiveScan(prefix, prefix, scan_op);

        cub::internal::ThreadScanExclusive(thread_data, thread_data, scan_op, prefix);

        if (threadIdx.y == 0){
            printf("Thread %d: %d %d \n", threadIdx.x, thread_data[0], thread_data[1]);
        }

        BlockStore(temp_storage.store).Store(result + x * blockDim.y * COLS, thread_data);
        __syncthreads();
    }
}

__global__ void reverse_warp_scan(int *input, int *result){

    typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT, BLOCK_SIZE.y> BlockStore;
    typedef WarpReverseScan<int, BLOCK_SIZE.x> WarpScan;

    if (threadIdx.x == 0 && threadIdx.y == 0){
        printf("Value of data %d \n", input[1]);
    }

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;


    constexpr int scans = ROWS / BLOCK_SIZE.y;
    for (int x = 0; x < scans; x ++){
        int thread_data[ITEMS];
        BlockLoad(temp_storage.load).Load(input + x * blockDim.y * COLS, thread_data);

        auto scan_op = cub::Sum();
        int prefix = ThreadReverseReduce(thread_data, scan_op);

        if (threadIdx.y == 0){
            printf("Thread %d: %d \n", threadIdx.x, prefix);
        }

        int warp_id = threadIdx.x / BLOCK_SIZE.x;
        int aggregate = 0;
        WarpScan().ExclusiveReverseScan(prefix, prefix, scan_op, aggregate);

        if (threadIdx.y == 0){
            printf("Thread %d: %d \n", threadIdx.x, prefix);
        }

        ThreadReverseScanExclusive(thread_data, thread_data, scan_op, prefix);

        if (threadIdx.y == 0){
            printf("Thread %d: %d %d \n", threadIdx.x, thread_data[0], thread_data[1]);
        }

        BlockStore(temp_storage.store).Store(result + x * blockDim.y * COLS, thread_data);
        __syncthreads();
    }
}

// -------------- 2D Warp Scan Kernels -------------------
__global__ void warp_scan_bidirectional(int *input, int *result){
    // Create classes for data-movement
    typedef cub::WarpLoad<int, ITEMS, cub::WARP_LOAD_DIRECT, BLOCK_SIZE.x> WarpLoad;
    typedef cub::WarpStore<int, ITEMS, cub::WARP_STORE_DIRECT, BLOCK_SIZE.x> WarpStore;
    typedef cub::WarpScan<int, BLOCK_SIZE.x> WarpScan;
    typedef WarpReverseScan<int, BLOCK_SIZE.x> ReverseWarpScan;


    // Create Shared Memory for communication
    __shared__ union {
        typename WarpLoad::TempStorage load[BLOCK_SIZE.y];
        typename WarpStore::TempStorage store[BLOCK_SIZE.y];
        typename WarpScan::TempStorage scan[BLOCK_SIZE.y];
    } temp_storage;

    // Create a Scan Op and Meta Info
    const int warp_id = threadIdx.y;
    auto scan_op = cub::Sum();


    // Iterate through chunks
    constexpr int chunks_x = COLS / (ITEMS * BLOCK_SIZE.x);
    constexpr int chunks_y = ROWS / BLOCK_SIZE.y;

    int thread_data[chunks_y][chunks_x][ITEMS];
    int forward_thread_data[chunks_y][chunks_x][ITEMS];
    for (int y = 0; y < chunks_y; y++){
        // Loop for Forward Prefix Sum
        BlockPrefixCallbackOp forward_prefix(0);
        for (int x = 0; x < chunks_x; x++){
            // Load the Data for the chunk
            WarpLoad(temp_storage.load[warp_id]).Load(input + y * COLS * BLOCK_SIZE.y + COLS * warp_id + x * ITEMS * BLOCK_SIZE.x, thread_data[y][x]);
            cub::CTA_SYNC();

            //Create objects for prefix sum
            auto warp_scan = WarpScan(temp_storage.scan[warp_id]);
            int block_aggregate;

            // Run a prefix sum for data
            int prefix = cub::internal::ThreadReduce(thread_data[y][x], scan_op); // Get Prefix 
            warp_scan.ExclusiveScan(prefix, prefix, scan_op, block_aggregate); // Run Scan on Prefix
            int block_prefix = forward_prefix(block_aggregate); // Get chunk prefix
            block_prefix = warp_scan.Broadcast(block_prefix, 0); // Broadcast from lane 0 to all others
            cub::CTA_SYNC();

            prefix = scan_op(prefix, block_prefix); // Update Prefix
            cub::internal::ThreadScanExclusive(thread_data[y][x], forward_thread_data[y][x], scan_op, prefix); // Run Local Exclusive Prefix
        }
        // Loop for Reverse Prefix Sum
        BlockPrefixCallbackOp reverse_postfix(0);
        for (int x = chunks_x - 1; x >= 0; x--){
            // Create objects for prefix sum
            auto reverse_warp_scan = ReverseWarpScan();
            int block_aggregate;

            // Run a reverse prefix sum for data
            int postfix = ThreadReverseReduce(thread_data[y][x], scan_op); // Get Postfix
            reverse_warp_scan.ExclusiveReverseScan(postfix, postfix, scan_op, block_aggregate); // Run Scan on Postfix
            int block_postfix = reverse_postfix(block_aggregate); // Get previous chunks value
            block_postfix = reverse_warp_scan.Broadcast(block_postfix, 0); 
            postfix = scan_op(postfix, block_postfix); // Update postfix
            ThreadReverseScanExclusive(thread_data[y][x], thread_data[y][x], scan_op, postfix);

            // Fuse Reverse with Forward
            for (int d = 0; d < ITEMS; d++){
                thread_data[y][x][d] += forward_thread_data[y][x][d];
            }

            // Store Data
            WarpStore(temp_storage.store[warp_id]).Store(result + y * COLS * BLOCK_SIZE.y + COLS * warp_id + x * ITEMS * BLOCK_SIZE.x, thread_data[y][x]);
            cub::CTA_SYNC();
        }
    }
}

__global__ void warp_scan2d(int *input, int *result){
    // Create classes for data-movement
    typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT, BLOCK_SIZE.y> BlockStore;
    typedef cub::BlockExchange<int, BLOCK_SIZE.x, ITEMS, false, BLOCK_SIZE.y> BlockExchange;
    typedef cub::WarpScan<int, BLOCK_SIZE.x> WarpScan;
    typedef WarpReverseScan<int, BLOCK_SIZE.x> ReverseWarpScan;


    // Create Shared Memory for communication
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockExchange::TempStorage exchange;
        typename WarpScan::TempStorage scan[BLOCK_SIZE.y];
    } temp_storage;

    // Create a Scan Op and Meta Info
    const int warp_id = threadIdx.x / BLOCK_SIZE.x;
    auto scan_op = cub::Sum();


    // Iterate through chunks
    constexpr int chunks_x = COLS / (ITEMS * BLOCK_SIZE.x);
    constexpr int chunks_y = ROWS / BLOCK_SIZE.y;

    if (threadIdx.x == 0 && threadIdx.y == 0){
        printf("%d %d \n", chunks_x, chunks_y);
    }
    int thread_data[chunks_y][chunks_x][ITEMS];
    int forward_thread_data[chunks_y][chunks_x][ITEMS];
    for (int y = 0; y < chunks_y; y++){
        // Loop for Forward Prefix Sum
        BlockPrefixCallbackOp forward_prefix(0);
        for (int x = 0; x < chunks_x; x++){
            // Load the Data for the chunk
            BlockLoad(temp_storage.load).Load(input + y * COLS * BLOCK_SIZE.y + x * ITEMS * BLOCK_SIZE.x, thread_data[y][x]);
            cub::CTA_SYNC();

            //Create objects for prefix sum
            auto warp_scan = WarpScan(temp_storage.scan[warp_id]);
            int block_aggregate;

            // Run a prefix sum for data
            int prefix = cub::internal::ThreadReduce(thread_data[y][x], scan_op); // Get Prefix 
            warp_scan.ExclusiveScan(prefix, prefix, scan_op, block_aggregate); // Run Scan on Prefix
            int block_prefix = forward_prefix(block_aggregate); // Get chunk prefix
            block_prefix = warp_scan.Broadcast(block_prefix, 0); // Broadcast from lane 0 to all others
            cub::CTA_SYNC();

            prefix = scan_op(prefix, block_prefix); // Update Prefix
            cub::internal::ThreadScanExclusive(thread_data[y][x], forward_thread_data[y][x], scan_op, prefix); // Run Local Exclusive Prefix
        }
        // Loop for Reverse Prefix Sum
        BlockPrefixCallbackOp reverse_postfix(0);
        for (int x = chunks_x - 1; x >= 0; x--){
            // Create objects for prefix sum
            auto reverse_warp_scan = ReverseWarpScan();
            int block_aggregate;

            // Run a reverse prefix sum for data
            int postfix = ThreadReverseReduce(thread_data[y][x], scan_op); // Get Postfix
            reverse_warp_scan.ExclusiveReverseScan(postfix, postfix, scan_op, block_aggregate); // Run Scan on Postfix
            int block_postfix = reverse_postfix(block_aggregate);
            block_postfix = reverse_warp_scan.Broadcast(block_postfix, 0); 
            postfix = scan_op(postfix, block_postfix); // Update postfix
            ThreadReverseScanExclusive(thread_data[y][x], thread_data[y][x], scan_op, postfix);

            // Fuse Reverse with Forward
            for (int d = 0; d < ITEMS; d++){
                thread_data[y][x][d] = forward_thread_data[y][x][d];
            }

            // Store Data
            BlockStore(temp_storage.store).Store(result + y * COLS * BLOCK_SIZE.y + x * ITEMS * BLOCK_SIZE.x, thread_data[y][x]);
            cub::CTA_SYNC();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0){
            printf("\n\n");
        }
    }

}


//
//
// __global__ void warp_scan2d(int *input, int *result){
//     typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y> BlockLoad;
//     typedef cub::BlockExchange<int, BLOCK_SIZE.x, ITEMS, false, BLOCK_SIZE.y> BlockExchange;
//     typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_TRANSPOSE, BLOCK_SIZE.y> BlockStore;
//     typedef cub::WarpScan<int, BLOCK_SIZE.x> WarpScan;
//
//     if (threadIdx.x == 0 && threadIdx.y == 0){
//         printf("Value of data %d \n", input[1]);
//     }
//
//     __shared__ union {
//         typename BlockLoad::TempStorage load;
//         typename WarpScan::TempStorage scan[BLOCK_SIZE.y];
//         typename BlockExchange::TempStorage exchange;
//         typename BlockStore::TempStorage store;
//     } temp_storage;
//
//     // Can Swap out the scan op for anything
//     auto scan_op = ScanOp();
//
//     // DATA NEEDS TO BE CHUNKED IN A ROW MAJOR ORDER
//     // constexpr int v_chunks_y = ROWS / BLOCK_SIZE.y;
//     // constexpr int v_chunks_x = COLS / (ITEMS * BLOCK_SIZE.x);
//     // int thread_data[v_chunks_y][v_chunks_x][ITEMS];
//     // for (int i = 0; i < v_chunks_y; i++){
//     //     for (int j = 0; j < v_chunks_x; j++){
//     //         // Load the data
//     //         BlockLoad(temp_storage.load).Load(input + i * COLS * BLOCK_SIZE.y + j * ITEMS * BLOCK_SIZE.x, thread_data[i][j]);
//     //
//     //         // Run the prefix sum
//     //     }
//     // }
//     // // Load the data
//     // BlockLoad(temp_storage.load).Load(input, thread_data);
//     // __syncthreads();
//     //
//     // // First prefix sum along the x-axis
//     // int prefix = cub::internal::ThreadReduce(thread_data, scan_op);
//     //
//     // int warp_id = threadIdx.x / BLOCK_SIZE.x;
//     // WarpScan(temp_storage.scan[warp_id]).ExclusiveSum(prefix, prefix);
//     // __syncthreads();
//     //
//     // cub::internal::ThreadScanExclusive(thread_data, thread_data, scan_op, prefix);
//     //
//     //
//     // // Rearrange the data (i.e.) transpose
//     // BlockExchange(temp_storage.exchange).BlockedToWarpStriped(thread_data, thread_data);
//     //
//     // // Second prefix sum along the y-axis
//     // prefix = cub::internal::ThreadReduce(thread_data, scan_op);
//     // WarpScan(temp_storage.scan[warp_id]).ExclusiveSum(prefix,prefix);
//     //
//     // cub::internal::ThreadScanExclusive(thread_data, thread_data, scan_op, prefix);
//     //
//     // BlockStore(temp_storage.store).Store(result, thread_data);
//     // __syncthreads();
// }