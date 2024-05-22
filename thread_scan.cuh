#pragma once


/**
 * Perform a sequential inclusive postfix reverse scan over the statically-sized \p input array, seeded with the specified \p postfix.  The aggregate is returned.
 * Fused with output, rather than overwriting it.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T FusedThreadReverseScanInclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    T inclusive = postfix;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(inclusive, input[i]);
        output[i] += inclusive;
    }
}

/**
 * Perform a sequential exclusive postfix reverse scan over the statically-sized \p input array, seeded with the specified \p postfix.  The aggregate is returned.
 * Fuse with output, rather than overwriting it.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T FusedThreadReverseScanExclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T postfix)
{
    // Careful, output maybe be aliased to input
    T exclusive = postfix;
    T inclusive;
    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; --i) {
        inclusive = scan_op(exclusive, input[i]);
        output[i] += exclusive;
        exclusive = inclusive;
    }
    return inclusive;
}



/**
 * Perform a sequential inclusive prefix scan over the statically-sized \p input array, seeded with the specified \p prefix.  The aggregate is returned.
 * Fused with output, rather than overwriting it.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T FusedThreadScanInclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T prefix)
{
    T inclusive = prefix;
    #pragma unroll
    for (int i = 0; i < LENGTH; ++i) {
        inclusive = scan_op(inclusive, input[i]);
        output[i] += inclusive;
    }
}

/**
 * Perform a sequential exclusive prefix scan over the statically-sized \p input array, seeded with the specified \p prefix.  The aggregate is returned.
 * Fuse with output, rather than overwriting it.
 */
template <
    int         LENGTH,
    typename    T,
    typename    ScanOp>
__device__ __forceinline__ T FusedThreadScanExclusive(
    const T (&input)[LENGTH],
    T (&output)[LENGTH],
    ScanOp scan_op,
    const T prefix)
{
    // Careful, output maybe be aliased to input
    T exclusive = prefix;
    T inclusive;
    #pragma unroll
    for (int i = 0; i < LENGTH; ++i) {
        inclusive = scan_op(exclusive, input[i]);
        output[i] += exclusive;
        exclusive = inclusive;
    }
    return inclusive;
}
