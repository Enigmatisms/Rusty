#pragma once
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_kernel.h"

__global__ void cuda_add_array_kernel(const float* const input, float* output, float v1, float v2, float v3);