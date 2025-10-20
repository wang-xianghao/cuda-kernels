#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <vector>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char *const func, const char *const file,
                const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char *const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void clear_cache()
{
    static int l2_size = 0;
    static unsigned char *tmp_data{nullptr};

    if (tmp_data == nullptr)
    {
        CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0));
        l2_size *= 2;
        CHECK_CUDA_ERROR(cudaMalloc(&tmp_data, l2_size));
    }

    CHECK_CUDA_ERROR(cudaMemset(tmp_data, 0, l2_size));
}

float measure_latency(std::function<void(cudaStream_t)> bound_function,
                      int num_repeats, cudaStream_t stream = 0)
{
    float latency;
    std::vector<float> latencies;

    // Create events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Measure latency
    for (int i = 0; i < num_repeats; ++i)
    {
        clear_cache();
        CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
        bound_function(stream);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&latency, start, stop));
        latencies.push_back(latency);
    }

    // Calculate total latency
    latency = std::reduce(latencies.begin(), latencies.end());

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return latency / num_repeats;
}

#endif // COMMON_UTILS_HPP