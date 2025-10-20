#include <cuda_runtime.h>
#include <random>
#include "common_utils.hpp"

// #define BLOCK_PATTERN

std::default_random_engine generator(69);
constexpr int ALPHABET_LENGTH = 26;

template <int NUM_ITERS>
__global__ void histogram_kernel(const char *data, int length, int *histo)
{

    __shared__ int histo_private[ALPHABET_LENGTH];

    for (int i = threadIdx.x; i < ALPHABET_LENGTH; i += blockDim.x)
    {
        histo_private[i] = 0;
    }
    __syncthreads();

    // Although using BLOCK_PATTERN naturally results in coalesced memory access,
    // a warp only reads 32bytes in a coalesced load since the data is char (1 byte),
    // which cannot fully utilize the global memory bandwidth.
    // Instead, the other pattern's loop can be unrollled by a factor of NUM_ITERS,
    // which yileds larger chunk of coelesced loads.
#ifdef BLOCK_PATTERN
    int idx = blockIdx.x * blockDim.x * NUM_ITERS + threadIdx.x;
    for (int i = idx; i < min(length, idx + NUM_ITERS * blockDim.x); i += blockDim.x)
    {
        int cid = data[i] - 'a';
        atomicAdd(histo_private + cid, 1);
    }
#else
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = NUM_ITERS * idx; i < min(length, (idx + 1) * NUM_ITERS); i += 1)
    {
        int cid = data[i] - 'a';
        atomicAdd(histo_private + cid, 1);
    }
#endif

    __syncthreads();

    for (int i = threadIdx.x; i < ALPHABET_LENGTH; i += blockDim.x)
    {
        int cnt = histo_private[i];
        atomicAdd(histo + i, cnt);
    }
}

void histogram(const char *data, int length, int *histo)
{
    // BLOCK_SIZE is a tradeoff between occupancy and stalling due to atomic operations
    //      - small: better occupancy
    //      - large: fewer atomic operations on slow global memory
    // We could achieve both advantages by coarsening.
    constexpr int BLOCK_SIZE = 512;
    constexpr int NUM_ITERS = 8;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(length, NUM_ITERS * BLOCK_SIZE));

    histogram_kernel<NUM_ITERS><<<gridDim, blockDim>>>(data, length, histo);
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