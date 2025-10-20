#include <cuda_runtime.h>
#include <random>
#include "common_utils.hpp"

std::default_random_engine generator(114514);

__global__ void sum_kernel(const float *input, int length, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length)
    {
        float x = input[tid];
        atomicAdd(output, x);
    }
}

void sum(const float *input, int length, float *output)
{
    constexpr int BLOCK_SIZE = 512;

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(length, BLOCK_SIZE));
    sum_kernel<<<gridDim, blockDim>>>(input, length, output);
    CHECK_LAST_CUDA_ERROR();
}

float initialize_data(float *input, int length)
{
    float sum = 0.0f;
    std::uniform_int_distribution<int> distrib(0, 5);
    for (int i = 0; i < length; ++i)
    {
        input[i] = static_cast<float>(distrib(generator));
        sum += input[i];
    }
    return sum;
}

int main()
{
    constexpr int length = 1024 * 1024 * 1024;
    constexpr int num_repeats = 8;

    // Alloate and initialize data on host
    float *input_host, *output_host, output_ref_host;
    CHECK_CUDA_ERROR(cudaMallocHost(&input_host, length * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&output_host, 1 * sizeof(float)));
    output_ref_host = initialize_data(input_host, length);

    // Allocate and copy to device
    float *input_device, *output_device;
    CHECK_CUDA_ERROR(cudaMalloc(&input_device, length * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&output_device, 1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(input_device, input_host, length * sizeof(float), cudaMemcpyHostToDevice));

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