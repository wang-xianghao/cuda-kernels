#include <cuda_runtime.h>
#include <random>
#include "common_utils.hpp"

std::default_random_engine generator(114514);

void initialize_data(int *A, int m)
{
    std::uniform_int_distribution<int> distrib(0, 5);
    for (int i = 0; i < m; ++i)
    {
        A[i] = distrib(generator);
    }

    for (int i = 1; i < m; ++i)
    {
        A[i] += A[i - 1];
    }
}

bool is_equal(const int *C, const int *C_ref, int k)
{
    for (int i = 0; i < k; ++i)
    {
        if (C[i] != C_ref[i])
        {
            printf("Incorrect result at %d: expect %d, but got %d\n", i, C_ref[i], C[i]);
            return false;
        }
    }
    return true;
}

__host__ __device__ void merge_sequential(const int *A, int m, const int *B, int n, int *C)
{
    int i = 0, j = 0, k = 0;
    while (i < m && j < n)
    {
        if (A[i] <= B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }
    while (i < m)
    {
        C[k++] = A[i++];
    }
    while (j < n)
    {
        C[k++] = B[j++];
    }
}

__device__ int co_rank(int k, const int *A, int m, const int *B, int n)
{
    int i = min(k, m);
    int j = k - i;
    int i_low = max(0, k - n);
    int j_low = max(0, k - m);
    int delta;

    while (true)
    {
        if (i > 0 && j < n && A[i - 1] > B[j])
        {
            delta = (i - i_low + 1) / 2;
            j_low = j;
            j += delta;
            i -= delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i])
        {
            delta = (j - j_low + 1) / 2;
            i_low = i;
            i += delta;
            j -= delta;
        }
        else
        {
            break;
        }
    }

    return i;
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void merge_kernel(const int *A, int m, const int *B, int n, int *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_cur = tid * COARSE_FACTOR;
    int k_next = min(k_cur + COARSE_FACTOR, m + n);
    int i_cur = co_rank(k_cur, A, m, B, n);
    int j_cur = k_cur - i_cur;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;
    merge_sequential(A + i_cur, i_next - i_cur, B + j_cur, j_next - j_cur, C + k_cur);
}

void merge(const int *A, int m, const int *B, int n, int *C)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int COARSE_FACTOR = 16;

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(m + n, COARSE_FACTOR * BLOCK_SIZE));
    merge_kernel<BLOCK_SIZE, COARSE_FACTOR><<<gridDim, blockDim>>>(A, m, B, n, C);
}

int main()
{
    constexpr int m = 1024 * 1024 * 128;
    constexpr int n = 1024 * 1024 * 256;
    constexpr int num_repeats = 8;

    // Alloate and initialize data on host
    int *A_host, *B_host, *C_host, *C_ref_host;
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B_host, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, (m + n) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_ref_host, (m + n) * sizeof(int)));
    initialize_data(A_host, m);
    initialize_data(B_host, n);
    merge_sequential(A_host, m, B_host, n, C_ref_host);

    // Allocate and copy to device
    int *A_device, *B_device, *C_device;
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&B_device, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, (m + n) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, n * sizeof(int), cudaMemcpyHostToDevice));

    // Run and verify results
    merge(A_device, m, B_device, n, C_device);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(C_host, C_device, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));
    bool correct = is_equal(C_host, C_ref_host, m + n);
    if (correct)
    {
        printf("Verified.\n");
    }
    printf("\n");

    // Measure performance
    float latency = measure_latency([&](cudaStream_t stream)
                                    { merge(A_device, m, B_device, n, C_device); }, num_repeats);
    float effective_bandwidth = (2 * m + 2 * n) * sizeof(int) * 1e-9 / (latency * 1e-3);
    printf("Latency: %.3f ms\n", latency);
    printf("Effective Bandwidth: %.3f GB/s\n", effective_bandwidth);

    return 0;
}