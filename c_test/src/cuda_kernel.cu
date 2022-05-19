#include "cuda_kernel.h"

__global__ void cuda_add_array_kernel(const float* const input, float* output, float v1, float v2, float v3) {
    const int tid = threadIdx.x, bid = blockIdx.x, tnum = blockDim.x;
    const int base_id = bid * tnum + tid;
    output[base_id] = output[base_id] + input[base_id] + v1 + v2 + v3;
}