#include "util.h"
#include "histogram.h"

__global__ void histogram_global_GPU(const uint8_t *gray_img, int* hist, int height, int width) {
    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    // global thread id and count
    int t = y * nx + x;
    int num_threads = nx * ny;
    
    // initialize hist in global memory
    for (int i = t; i < Y_LEVELS; i += num_threads)
        hist[i] = 0;

    for (int i = y; i < height; i += ny)
        for (int j = x; j < width; j += nx)
            atomicAdd(hist + gray_img[i * width + j], 1);
}

__global__ void histogram_local_globalmem_GPU(const uint8_t *gray_img, int* all_hists, int height, int width) {
    int t = threadIdx.y * blockDim.x + threadIdx.x; // thread linear idx in block
    int num_threads = blockDim.x * blockDim.y;      // thread total count in block
    
    // initialize local hist in global memory
    all_hists += (blockIdx.y * gridDim.x + blockIdx.x) * Y_LEVELS;
    for (int i = t; i < Y_LEVELS; i += num_threads)
        all_hists[i] = 0;

    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    for (int i = y; i < height; i += ny)
        for (int j = x; j < width; j += nx)
            atomicAdd(all_hists + gray_img[i * width + j], 1);
}

__global__ void histogram_local_sharedmem_GPU(const uint8_t *gray_img, int* all_hists, int height, int width) {
    int t = threadIdx.y * blockDim.x + threadIdx.x; // thread linear idx in block
    int num_threads = blockDim.x * blockDim.y;      // thread total count in block
    
    // initialize local histogram for one block
    __shared__ int local_hist[Y_LEVELS];
    for (int i = t; i < Y_LEVELS; i += num_threads)
        local_hist[i] = 0;
    __syncthreads();

    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    for (int i = y; i < height; i += ny)
        for (int j = x; j < width; j += nx)
            atomicAdd(local_hist + gray_img[i * width + j], 1);
    __syncthreads();

    // copy local hist to global memory
    all_hists += (blockIdx.y * gridDim.x + blockIdx.x) * Y_LEVELS;
    for (int i = t; i < Y_LEVELS; i += num_threads)
        all_hists[i] = local_hist[i];
}

__global__ void histogram_finalize_GPU(const int *all_hists, int *hist, int num_hists) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; // thread global idx
    if (t < Y_LEVELS) {
        int total = 0;
        for (int i = 0; i < num_hists; i++) 
            total += all_hists[i * Y_LEVELS + t];
        hist[t] = total;
    }
}
