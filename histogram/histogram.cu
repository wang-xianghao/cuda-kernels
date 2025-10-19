#include <cuda_runtime.h>
#include <random>
#include "common_utils.hpp"

std::default_random_engine generator(69);
constexpr int ALPHABET_LENGTH = 26;

__global__ void histogram_kernel(const char *data, int length, int *histo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int histo_private[ALPHABET_LENGTH];

    for (int i = threadIdx.x; i < ALPHABET_LENGTH; i += blockDim.x)
    {
        histo_private[i] = 0;
    }
    __syncthreads();

    if (idx < length)
    {
        int cid = data[idx] - 'a';
        atomicAdd(histo_private + cid, 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ALPHABET_LENGTH; i += blockDim.x)
    {
        int cnt = histo_private[i];
        atomicAdd(histo + i, cnt);
    }
}

void histogram(const char *data, int length, int *histo)
{
    constexpr int BLOCK_SIZE = 1024;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(length, BLOCK_SIZE));

    histogram_kernel<<<gridDim, blockDim>>>(data, length, histo);
    CHECK_LAST_CUDA_ERROR();
}

void initialize_data(char *data, int length)
{
    std::uniform_int_distribution<> distrib(0, 25);
    for (int i = 0; i < length; ++i)
    {
        data[i] = static_cast<char>(distrib(generator) + 'a');
    }
}

int main()
{
    constexpr int length = 1024 * 1024 * 1024;
    constexpr int num_repeats = 8;

    // Allocate and initialize host data
    char *data_host;
    int *histo_host;
    CHECK_CUDA_ERROR(cudaMallocHost(&data_host, length * sizeof(char)));
    CHECK_CUDA_ERROR(cudaMallocHost(&histo_host, ALPHABET_LENGTH * sizeof(int)));
    initialize_data(data_host, length);

    // Copy data to device
    char *data_device;
    int *histo_device;
    CHECK_CUDA_ERROR(cudaMalloc(&data_device, length * sizeof(char)));
    CHECK_CUDA_ERROR(cudaMalloc(&histo_device, ALPHABET_LENGTH * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_device, data_host, length * sizeof(char), cudaMemcpyHostToDevice));

    // Run and check results
    histogram(data_device, length, histo_device);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(histo_host, histo_device, ALPHABET_LENGTH * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < ALPHABET_LENGTH; ++i)
    {
        printf("%c(%6.3f%) ", i + 'a', 100 * static_cast<float>(histo_host[i]) / length);
        if ((i + 1) % 13 == 0)
        {
            printf("\n");
        }
    }
    printf("\n");

    // Measure performance
    float latency = measure_latency([&](cudaStream_t stream)
                                    { histogram(data_device, length, histo_device); }, num_repeats);
    float effective_bandwidth = (26 + length) * sizeof(char) * 1e-9 / (latency * 1e-3);
    printf("Latency: %.3f ms\n", latency);
    printf("Effective Bandwidth: %.3f GB/s\n", effective_bandwidth);

    return 0;
}