
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

        BlockStore(temp_storage.store).Store(result + x * blockDim.y * COLS, thread_data);
        __syncthreads();
    }
}

__global__ void shared_warp_scan(int *input, int*result){
    typedef cub::BlockLoad<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_LOAD_DIRECT, BLOCK_SIZE.y> BlockLoad;
    typedef cub::BlockStore<int, BLOCK_SIZE.x, ITEMS, cub::BLOCK_STORE_DIRECT, BLOCK_SIZE.y> BlockStore;
    typedef cub::WarpScan<int, BLOCK_SIZE.x> WarpScan;

    // Define memory layout
    union SmemLayout{
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename WarpScan:: TempStorage scan[BLOCK_SIZE.y];
    };

    extern __shared__ __align__ (alignof(SmemLayout)) char smem[];

    // reinterpret mem
    auto& load_mem = reinterpret_cast<BlockLoad::TempStorage&>(smem);
    auto& store_mem = reinterpret_cast<BlockStore::TempStorage&>(smem);
    auto* scan_mem = reinterpret_cast<WarpScan::TempStorage*>(smem);

    constexpr int scans = ROWS / BLOCK_SIZE.y;
    for (int x = 0; x < scans; x ++){
        int thread_data[ITEMS];
        BlockLoad(load_mem).Load(input + x * blockDim.y * COLS, thread_data);
        __syncthreads();

        auto scan_op = cub::Sum();
        int prefix = cub::internal::ThreadReduce(thread_data, scan_op);

        int warp_id = threadIdx.x / BLOCK_SIZE.x;
        WarpScan(scan_mem[warp_id]).ExclusiveScan(prefix, prefix, scan_op);
        __syncthreads();

        cub::internal::ThreadScanExclusive(thread_data, thread_data, scan_op, prefix);

        BlockStore(store_mem).Store(result + x * blockDim.y * COLS, thread_data);
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

__global__ void warp_scan_orthoganal_2d(int *input, int *result, int *result2){
    // Create classes for data-movement
    typedef cub::WarpLoad<int, ITEMS, cub::WARP_LOAD_DIRECT, BLOCK_SIZE.x> WarpLoad;
    typedef cub::WarpStore<int, ITEMS, cub::WARP_STORE_DIRECT, BLOCK_SIZE.x> WarpStore;
    typedef cub::BlockExchange<int, BLOCK_SIZE.x, ITEMS, false, BLOCK_SIZE.y> BlockExchange;
    typedef cub::WarpScan<int, BLOCK_SIZE.x> HorizontalWarpScan;
    typedef WarpReverseScan<int, BLOCK_SIZE.x> HorizontalReverseWarpScan;
    typedef cub::WarpScan<int, (int)(BLOCK_SIZE.y / ITEMS)> VerticalWarpScan;
    typedef WarpReverseScan<int, (int)(BLOCK_SIZE.y / ITEMS)> VerticalReverseWarpScan;

    input = input + (blockIdx.x * ROWS * COLS);
    result = result + (blockIdx.x * ROWS * COLS);
    result2 = result2 + (blockIdx.x * ROWS * COLS);


    // Create Shared Memory for communication
    __shared__ union {
        typename WarpLoad::TempStorage load[BLOCK_SIZE.y];
        typename WarpStore::TempStorage store[BLOCK_SIZE.y];
        typename BlockExchange::TempStorage exchange;
        typename HorizontalWarpScan::TempStorage horizontal_scan[BLOCK_SIZE.y];
        typename VerticalWarpScan::TempStorage vertical_scan[(int)(BLOCK_SIZE.y / ITEMS)];
    } temp_storage;

    // Create a Scan Op and Meta Info
    const int horizontal_warp_id = threadIdx.y;
    const int vertical_warp_id = threadIdx.y / ITEMS;
    auto scan_op = cub::Sum();

    // Get Chunk Info
    constexpr int chunks_x = COLS / (ITEMS * BLOCK_SIZE.x);
    constexpr int chunks_y = ROWS / BLOCK_SIZE.y;

    // Create storage (Should create better methods to fuse these)
    int thread_data[chunks_y][chunks_x][ITEMS];
    int forward_thread_data[chunks_y][chunks_x][ITEMS];
    int reverse_thread_data[chunks_y][chunks_x][ITEMS];
    // LEFT-RIGHT BIDIRECTIONAL SCAN
    for (int y = 0; y < chunks_y; y++){
        // Loop for Forward Prefix Sum
        BlockPrefixCallbackOp forward_prefix(0);
        for (int x = 0; x < chunks_x; x++){
            // Load the Data for the chunk
            WarpLoad(temp_storage.load[horizontal_warp_id]).Load(input + y * COLS * BLOCK_SIZE.y + COLS * horizontal_warp_id + x * ITEMS * BLOCK_SIZE.x, thread_data[y][x]);
            cub::CTA_SYNC();

            //Create objects for prefix sum
            auto warp_scan = HorizontalWarpScan(temp_storage.horizontal_scan[horizontal_warp_id]);
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
            // Create objects for postfix sum
            auto reverse_warp_scan = HorizontalReverseWarpScan();
            int block_aggregate;

            // Run a reverse prefix sum for data
            int postfix = ThreadReverseReduce(thread_data[y][x], scan_op); // Get Postfix
            reverse_warp_scan.ExclusiveReverseScan(postfix, postfix, scan_op, block_aggregate); // Run Scan on Postfix
            int block_postfix = reverse_postfix(block_aggregate); // Get previous chunks value
            block_postfix = reverse_warp_scan.Broadcast(block_postfix, 0); 
            postfix = scan_op(postfix, block_postfix); // Update postfix
            ThreadReverseScanExclusive(thread_data[y][x], reverse_thread_data[y][x], scan_op, postfix);

            // Fuse Reverse with Forward
            for (int d = 0; d < ITEMS; d++){
                forward_thread_data[y][x][d] += reverse_thread_data[y][x][d];
            }

            // Store Data
            WarpStore(temp_storage.store[horizontal_warp_id]).Store(result + y * COLS * BLOCK_SIZE.y + COLS * horizontal_warp_id + x * ITEMS * BLOCK_SIZE.x, forward_thread_data[y][x]);
            cub::CTA_SYNC();
        }
    }

    int down_thread_data[chunks_y][chunks_x][ITEMS];
    int up_thread_data[chunks_y][chunks_x][ITEMS];
    // UP-DOWN BIDIRECTIONAL SCAN
    for (int x = 0; x < chunks_x; x++){
        // Loop through Downward Prefix Sum
        BlockPrefixCallbackOp down_prefix(0);
        for (int y = 0; y < chunks_y; y++){
            // Rearrange data accross the block
            int thread_rank[ITEMS];
            for (int i = 0; i < ITEMS; i++){ // If calling this multiple times then cache it
                thread_rank[i] = threadIdx.y + (i + threadIdx.x * ITEMS) * BLOCK_SIZE.y; // Essentially just transposing our data
            }
            BlockExchange(temp_storage.exchange).ScatterToBlocked(thread_data[y][x], thread_data[y][x], thread_rank); // Transposing accross threads

            // Create objects for prefix sum
            auto warp_scan = VerticalWarpScan(temp_storage.vertical_scan[vertical_warp_id]);
            int block_aggregate;

            // Run a prefix sum for data
            int prefix = cub::internal::ThreadReduce(thread_data[y][x], scan_op); // Get Prefix 
            warp_scan.ExclusiveScan(prefix, prefix, scan_op, block_aggregate); // Run Scan on Prefix
            int block_prefix = down_prefix(block_aggregate); // Get chunk prefix
            __syncthreads();
            block_prefix = warp_scan.Broadcast(block_prefix, 0); // Broadcast from lane 0 to all others
            cub::CTA_SYNC();

            prefix = scan_op(prefix, block_prefix); // Update Prefix
            cub::internal::ThreadScanExclusive(thread_data[y][x], down_thread_data[y][x], scan_op, prefix); // Run Local Exclusive Prefix
        }

        // Loop through Upwards Prefix Sum
        BlockPrefixCallbackOp up_postfix(0);
        for (int y = chunks_y - 1; y >= 0; y--){
            // Create objects for postfix sum
            auto reverse_warp_scan = VerticalReverseWarpScan();
            int block_aggregate;

            // Run a reverse prefix sum for data
            int postfix = ThreadReverseReduce(thread_data[y][x], scan_op); // Get Postfix
            reverse_warp_scan.ExclusiveReverseScan(postfix, postfix, scan_op, block_aggregate); // Run Scan on Postfix
            int block_postfix = up_postfix(block_aggregate); // Get previous chunks value
            block_postfix = reverse_warp_scan.Broadcast(block_postfix, 0); 
            postfix = scan_op(postfix, block_postfix); // Update postfix
            ThreadReverseScanExclusive(thread_data[y][x], up_thread_data[y][x], scan_op, postfix);

            // Fuse Reverse with Forward
            for (int d = 0; d < ITEMS; d++){
                down_thread_data[y][x][d] += up_thread_data[y][x][d];
            }
            int thread_rank[ITEMS];
            // Need to test if non symmetric matrices are worth it, (can simplify calculation if not)
            constexpr int warp_size_vertical = BLOCK_SIZE.y / ITEMS;
            const int idx = threadIdx.x + threadIdx.y * BLOCK_SIZE.x;
            for (int i = 0; i < ITEMS; i++){ // If calling this multiple times then cache it
                thread_rank[i] = (idx / warp_size_vertical) + (idx % warp_size_vertical * ITEMS + i) * (BLOCK_SIZE.x * ITEMS); // Essentially just reverse transposing our data (slighly weird bc of thread arrangement)
            }

            BlockExchange(temp_storage.exchange).ScatterToBlocked(down_thread_data[y][x], down_thread_data[y][x], thread_rank); // Transposing accross threads
            // Store Data
            WarpStore(temp_storage.store[horizontal_warp_id]).Store(result2 + y * COLS * BLOCK_SIZE.y + COLS * horizontal_warp_id + x * ITEMS * BLOCK_SIZE.x, down_thread_data[y][x]);
            cub::CTA_SYNC();
        }
    }
}




// __global__ void warp_scan_orthoganal_2d_shared(int *input, int *result, int *result2){
//     // Create classes for data-movement
//     typedef cub::WarpLoad<int, ITEMS, cub::WARP_LOAD_DIRECT, BLOCK_SIZE.x> WarpLoad;
//     typedef cub::WarpStore<int, ITEMS, cub::WARP_STORE_DIRECT, BLOCK_SIZE.x> WarpStore;
//     typedef cub::BlockExchange<int, BLOCK_SIZE.x, ITEMS, false, BLOCK_SIZE.y> BlockExchange;
//     typedef cub::WarpScan<int, BLOCK_SIZE.x> HorizontalWarpScan;
//     typedef WarpReverseScan<int, BLOCK_SIZE.x> HorizontalReverseWarpScan;
//     typedef cub::WarpScan<int, (int)(BLOCK_SIZE.y / ITEMS)> VerticalWarpScan;
//     typedef WarpReverseScan<int, (int)(BLOCK_SIZE.y / ITEMS)> VerticalReverseWarpScan;
//
//     union SmemLayout{
//         typename WarpLoad::TempStorage load[BLOCK_SIZE.y];
//         typename WarpStore::TempStorage store[BLOCK_SIZE.y];
//         typename BlockExchange::TempStorage exchange;
//         typename HorizontalWarpScan::TempStorage h_scan[BLOCK_SIZE.y];
//         typename VerticalWarpScan::TempStorage v_scan[(int)(BLOCK_SIZE.y / ITEMS)];
//     };
//
//     // Define memory layout
//     extern __shared__ __align__ (alignof(SmemLayout)) char smem[];
//
//     // reinterpret mem
//     auto* load_mem = reinterpret_cast<WarpLoad::TempStorage*>(smem);
//     auto* store_mem = reinterpret_cast<WarpStore::TempStorage*>(smem);
//     auto& exchange_mem = reinterpret_cast<BlockExchange::TempStorage&>(smem);
//     auto* h_scan_mem = reinterpret_cast<HorizontalWarpScan::TempStorage*>(smem);
//     auto* v_scan_mem = reinterpret_cast<VerticalWarpScan::TempStorage*>(smem);
//
//
//     // Create Shared Memory for communication
//
//     // Create a Scan Op and Meta Info
//     const int horizontal_warp_id = threadIdx.y;
//     const int vertical_warp_id = threadIdx.y / ITEMS;
//     auto scan_op = cub::Sum();
//
//     // Get Chunk Info
//     constexpr int chunks_x = COLS / (ITEMS * BLOCK_SIZE.x);
//     constexpr int chunks_y = ROWS / BLOCK_SIZE.y;
//
//     // Create storage (Should create better methods to fuse these)
//     int thread_data[chunks_y][chunks_x][ITEMS];
//     int forward_thread_data[chunks_y][chunks_x][ITEMS];
//     int reverse_thread_data[chunks_y][chunks_x][ITEMS];
//     // LEFT-RIGHT BIDIRECTIONAL SCAN
//     for (int y = 0; y < chunks_y; y++){
//         // Loop for Forward Prefix Sum
//         BlockPrefixCallbackOp forward_prefix(0);
//         for (int x = 0; x < chunks_x; x++){
//             // Load the Data for the chunk
//             __syncthreads();
//             WarpLoad(load_mem[horizontal_warp_id]).Load(input + y * COLS * BLOCK_SIZE.y + COLS * horizontal_warp_id + x * ITEMS * BLOCK_SIZE.x, thread_data[y][x]);
//
//             // Run a prefix sum for data
//             int block_aggregate;
//             int prefix = cub::internal::ThreadReduce(thread_data[y][x], scan_op); // Get Prefix 
//             __syncthreads();
//             HorizontalWarpScan(h_scan_mem[horizontal_warp_id]).ExclusiveScan(prefix, prefix, scan_op, block_aggregate); // Run Scan on Prefix
//
//             int block_prefix = forward_prefix(block_aggregate); // Get chunk prefix
//             __syncthreads();
//             block_prefix = HorizontalWarpScan(h_scan_mem[horizontal_warp_id]).Broadcast(block_prefix, 0); // Broadcast from lane 0 to all others
//
//             prefix = scan_op(prefix, block_prefix); // Update Prefix
//             cub::internal::ThreadScanExclusive(thread_data[y][x], forward_thread_data[y][x], scan_op, prefix); // Run Local Exclusive Prefix
//         }
//         // Loop for Reverse Prefix Sum
//         BlockPrefixCallbackOp reverse_postfix(0);
//         for (int x = chunks_x - 1; x >= 0; x--){
//             // Create objects for postfix sum
//
//             // Run a reverse prefix sum for data
//             int block_aggregate;
//             int postfix = ThreadReverseReduce(thread_data[y][x], scan_op); // Get Postfix
//             HorizontalReverseWarpScan().ExclusiveReverseScan(postfix, postfix, scan_op, block_aggregate); // Run Scan on Postfix
//
//             int block_postfix = reverse_postfix(block_aggregate); // Get previous chunks value
//             block_postfix = HorizontalReverseWarpScan().Broadcast(block_postfix, 0); 
//             postfix = scan_op(postfix, block_postfix); // Update postfix
//             ThreadReverseScanExclusive(thread_data[y][x], reverse_thread_data[y][x], scan_op, postfix);
//
//             // Fuse Reverse with Forward
//             for (int d = 0; d < ITEMS; d++){
//                 forward_thread_data[y][x][d] += reverse_thread_data[y][x][d];
//             }
//
//             // Store Data
//             __syncthreads();
//             WarpStore(store_mem[horizontal_warp_id]).Store(result + y * COLS * BLOCK_SIZE.y + COLS * horizontal_warp_id + x * ITEMS * BLOCK_SIZE.x, forward_thread_data[y][x]);
//         }
//     }
//
//     int down_thread_data[chunks_y][chunks_x][ITEMS];
//     int up_thread_data[chunks_y][chunks_x][ITEMS];
//     // UP-DOWN BIDIRECTIONAL SCAN
//     for (int x = 0; x < chunks_x; x++){
//         // Loop through Downward Prefix Sum
//         BlockPrefixCallbackOp down_prefix(0);
//         for (int y = 0; y < chunks_y; y++){
//             // Rearrange data accross the block
//             int thread_rank[ITEMS];
//             for (int i = 0; i < ITEMS; i++){ // If calling this multiple times then cache it
//                 thread_rank[i] = threadIdx.y + (i + threadIdx.x * ITEMS) * BLOCK_SIZE.y; // Essentially just transposing our data
//             }
//             __syncthreads();
//             BlockExchange(exchange_mem).ScatterToBlocked(thread_data[y][x], thread_data[y][x], thread_rank); // Transposing accross threads
//
//             // Run a prefix sum for data
//             int block_aggregate;
//             int prefix = cub::internal::ThreadReduce(thread_data[y][x], scan_op); // Get Prefix 
//             __syncthreads();
//             VerticalWarpScan(v_scan_mem[vertical_warp_id]).ExclusiveScan(prefix, prefix, scan_op, block_aggregate); // Run Scan on Prefix
//
//
//             int block_prefix = down_prefix(block_aggregate); // Get chunk prefix
//             __syncthreads();
//             block_prefix = VerticalWarpScan(v_scan_mem[vertical_warp_id]).Broadcast(block_prefix, 0); // Broadcast from lane 0 to all others
//
//             prefix = scan_op(prefix, block_prefix); // Update Prefix
//             cub::internal::ThreadScanExclusive(thread_data[y][x], down_thread_data[y][x], scan_op, prefix); // Run Local Exclusive Prefix
//         }
//
//         // Loop through Upwards Prefix Sum
//         BlockPrefixCallbackOp up_postfix(0);
//         for (int y = chunks_y - 1; y >= 0; y--){
//             // Run a reverse prefix sum for data
//             int block_aggregate;
//             int postfix = ThreadReverseReduce(thread_data[y][x], scan_op); // Get Postfix
//             VerticalReverseWarpScan().ExclusiveReverseScan(postfix, postfix, scan_op, block_aggregate); // Run Scan on Postfix
//             int block_postfix = up_postfix(block_aggregate); // Get previous chunks value
//
//             block_postfix = VerticalReverseWarpScan().Broadcast(block_postfix, 0); 
//             postfix = scan_op(postfix, block_postfix); // Update postfix
//             ThreadReverseScanExclusive(thread_data[y][x], up_thread_data[y][x], scan_op, postfix);
//
//             // Fuse Reverse with Forward
//             for (int d = 0; d < ITEMS; d++){
//                 down_thread_data[y][x][d] += up_thread_data[y][x][d];
//             }
//             int thread_rank[ITEMS];
//             // Need to test if non symmetric matrices are worth it, (can simplify calculation if not)
//             constexpr int warp_size_vertical = BLOCK_SIZE.y / ITEMS;
//             const int idx = threadIdx.x + threadIdx.y * BLOCK_SIZE.x;
//             for (int i = 0; i < ITEMS; i++){ // If calling this multiple times then cache it
//                 thread_rank[i] = (idx / warp_size_vertical) + (idx % warp_size_vertical * ITEMS + i) * (BLOCK_SIZE.x * ITEMS); // Essentially just reverse transposing our data (slighly weird bc of thread arrangement)
//             }
//
//             __syncthreads();
//             BlockExchange(exchange_mem).ScatterToBlocked(down_thread_data[y][x], down_thread_data[y][x], thread_rank); // Transposing accross threads
//
//             // Store Data
//             __syncthreads();
//             WarpStore(store_mem[horizontal_warp_id]).Store(result2 + y * COLS * BLOCK_SIZE.y + COLS * horizontal_warp_id + x * ITEMS * BLOCK_SIZE.x, down_thread_data[y][x]);
//             cub::CTA_SYNC();
//         }
//     }
// }
