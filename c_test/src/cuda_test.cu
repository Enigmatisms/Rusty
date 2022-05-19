#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "cuda_kernel.h"
#include "cuda_err_check.hpp"

struct Vec4 {
    float x, y, z;
    int num;
};

extern  "C" {
void cuda_add_array(const float* const input, float* output, Vec4 vec4) {
    printf("%f, %f, %f, %d\n", vec4.x, vec4.y, vec4.z, vec4.num);
    int block_num = vec4.num >> 8;
    float* input_ptr, *output_ptr;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &input_ptr, vec4.num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &output_ptr, vec4.num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(input_ptr, input, vec4.num * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(output_ptr, output, vec4.num * sizeof(float), cudaMemcpyHostToDevice));
    cuda_add_array_kernel<<<block_num, 256>>>(input_ptr, output_ptr, vec4.x, vec4.y, vec4.z);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaMemcpy(output, output_ptr, vec4.num * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(input_ptr));
    CUDA_CHECK_RETURN(cudaFree(output_ptr));
}
}
