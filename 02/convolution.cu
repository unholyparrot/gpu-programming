#include <fstream>
#include <iostream>

#include "util.h"
#include "convolution.h"

__global__ void conv2d(const float* input, float* output, const float* kernel, int height, int width, int channels) {
    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int nz = blockDim.z * gridDim.z;
    // global thread id and count
    int t = z * nx * ny + y * nx + x;
    int num_threads = nx * ny * nz;

    // loop replaces if statement, in most cases calculated once
    for (int out_y = y; out_y < height; out_y += ny) {
        for (int out_x = x; out_x < width; out_x += nx) {
            // loop for current elem calculation
            float result = 0.0f;
            for (int i = 0; i < KERNEL_SIZE; ++i) {
                for (int j = 0; j < KERNEL_SIZE; ++j) {
                    int in_y = CLIP(out_y + i - KERNEL_HALF, 0, height - 1);
                    int in_x = CLIP(out_x + j - KERNEL_HALF, 0, width - 1);
                    result += input[in_y * width + in_x] * kernel[i * KERNEL_SIZE + j];
                }
            }
            output[out_y * width + out_x] = result;
        }
    }
}

void Conv2d::load_weights(const char* fname) {
    int num_weights = out_ch * in_ch * KERNEL_SIZE * KERNEL_SIZE;
    std::ifstream ifile(fname, std::ios::binary);
    if (!ifile.is_open()) {
        std::cerr << "File with weights do not exist: " << fname << std::endl;
    }
    ifile.read((char*)weights_host, num_weights * sizeof(float));
    cudaMemcpy(weights_device, weights_host,  num_weights * sizeof(float), cudaMemcpyHostToDevice);
    ifile.close();
}

void Conv2d::forward(const float* input_device, float* output_device, int height, int width) const {
    dim3 grid_dim((width + BLOCK_SZ_2D - 1) / BLOCK_SZ_2D, (height + BLOCK_SZ_2D - 1) / BLOCK_SZ_2D);
    dim3 block_dim(BLOCK_SZ_2D, BLOCK_SZ_2D);
    int one_filter_size = in_ch * KERNEL_SIZE * KERNEL_SIZE;
    int feature_map_size = height * width;
    for (int i = 0; i < out_ch; ++i) {
        conv2d<<<grid_dim, block_dim>>>(
            input_device, 
            output_device + i * feature_map_size, 
            weights_device + i * one_filter_size,
            height, width, in_ch
        );
    }
}

void Conv2d::forward_transpose(const float* input_device, float* output_device, int height, int width) const {
    // dim3 grid_dim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    // int one_filter_size = in_ch * KERNEL_SIZE * KERNEL_SIZE;
    // int feature_map_size = height * width;
    // for (int i = 0; i < out_ch; ++i) {
    //     conv2d<<<grid_dim, block_dim>>>(
    //         input_device, 
    //         output_device + i * feature_map_size, 
    //         weights_device + i * one_filter_size,
    //         height, width, in_ch
    //     );
    // }
}