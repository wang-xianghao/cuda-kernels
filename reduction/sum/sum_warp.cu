#include <cuda_runtime.h>
#include <random>
#include "common_utils.hpp"

std::default_random_engine generator(114514);

template <int BLOCK_SIZE, int WARP_SIZE>
__global__ void sum_kernel(float *input, int length, float *output)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ float input_s[BLOCK_SIZE / WARP_SIZE];
    float val = idx < length ? input[idx] : 0.0f;

    // Phase 1
    for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }
    if (lane_id == 0)
    {
        input_s[warp_id] = val;
    }
    __syncthreads();

    // Phase 2
    if (warp_id == 0)
    {
        val = lane_id < BLOCK_SIZE / WARP_SIZE ? input_s[lane_id] : 0.0f;
        for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2)
        {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }

        if (lane_id == 0)
        {
            atomicAdd(output, val);
        }
    }
}

void sum(float *input, int length, float *output)
{
    constexpr int BLOCK_SIZE = 1024;
    constexpr int WARP_SIZE = 32;

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(length, BLOCK_SIZE));
    sum_kernel<BLOCK_SIZE, WARP_SIZE><<<gridDim, blockDim>>>(input, length, output);
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