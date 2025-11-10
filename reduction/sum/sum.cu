#include <cuda_runtime.h>
#include <random>
#include "common_utils.hpp"

std::default_random_engine generator(114514);

template <int WARP_SIZE>
__device__ float warp_sum(float val)
{
    for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }

    return val;
}

template <int BLOCK_SIZE, int COARSE_FACTOR, int WARP_SIZE>
__global__ void sum_kernel(float *input, int length, float *output)
{
    int offset = COARSE_FACTOR * blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x;
    int idx = offset + local_idx;
    __shared__ float sums[BLOCK_SIZE];

    // Store local reduction result into the shared memory
    float sum = idx < length ? input[idx] : 0.0f;
    for (int i = idx + BLOCK_SIZE; i < min(offset + COARSE_FACTOR * BLOCK_SIZE, length); i += BLOCK_SIZE)
    {
        sum += input[i];
    }
    sums[local_idx] = sum;

    for (int stride = BLOCK_SIZE / 2; stride >= WARP_SIZE; stride /= 2)
    {
        __syncthreads();
        if (local_idx < stride)
        {
            sums[local_idx] += sums[local_idx + stride];
        }
    }

    __syncthreads();
    if (local_idx < WARP_SIZE)
    {
        sum = sums[local_idx];
        sum = warp_sum<WARP_SIZE>(sum);
    }

    if (local_idx == 0)
    {
        atomicAdd(output, sum);
    }
}

void sum(float *input, int length, float *output)
{
    constexpr int BLOCK_SIZE = 1024;
    constexpr int COARSE_FACTOR = 8;
    constexpr int WARP_SIZE = 32;

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(length, BLOCK_SIZE * COARSE_FACTOR));
    sum_kernel<BLOCK_SIZE, COARSE_FACTOR, WARP_SIZE><<<gridDim, blockDim>>>(input, length, output);
    CHECK_LAST_CUDA_ERROR();
}

float initialize_data(float *input, int length)
{
    float sum = 0.0f;
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);
    for (int i = 0; i < length; ++i)
    {
        input[i] = distrib(generator);
        sum += input[i];
    }
    return sum;
}

int main()
{
    constexpr int length = 1024 * 1024 * 512;
    constexpr int num_repeats = 8;

    // Alloate and initialize data on host
    float *input_host, *output_host, output_ref_host;
    CHECK_CUDA_ERROR(cudaMallocHost(&input_host, length * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&output_host, 1 * sizeof(float)));
    output_ref_host = initialize_data(input_host, length);
    *output_host = 0.0f;

    // Allocate and copy to device
    float *input_device, *output_device;
    CHECK_CUDA_ERROR(cudaMalloc(&input_device, length * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&output_device, 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(input_device, input_host, length * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(output_device, output_host, 1 * sizeof(float), cudaMemcpyHostToDevice));

    // Run and check results
    sum(input_device, length, output_device);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(output_host, output_device, 1 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum: %.3f\n", *output_host);
    printf("Sum (Ref): %.3f\n", output_ref_host);
    printf("Difference: %.3e\n", abs(*output_host - output_ref_host));
    printf("\n");

    // Measure performance
    float latency = measure_latency([&](cudaStream_t stream)
                                    { sum(input_device, length, output_device); }, num_repeats);
    float effective_bandwidth = length * sizeof(float) * 1e-9 / (latency * 1e-3);
    float effective_tflops = length * 1e-12 / (latency * 1e-3);
    printf("Latency: %.3f ms\n", latency);
    printf("Effective Bandwidth: %.3f GB/s\n", effective_bandwidth);
    printf("Effective TFLOPS: %.3f TFLOPS\n", effective_tflops);

    return 0;
}