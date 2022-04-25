#include <fstream>
#include <iostream>

#include "util.h"
#include "convolution.h"

__global__ void conv2d(const float* input, float* output, const float* kernel, float bias, int activation_num, int height, int width, int channels) {
    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // loop replaces if statement, in most cases calculated once
    for (int out_y = y; out_y < height; out_y += ny) {
        for (int out_x = x; out_x < width; out_x += nx) {
            // loop for current elem calculation
            float result = 0.0f;
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < KERNEL_SIZE; ++i) {
                    for (int j = 0; j < KERNEL_SIZE; ++j) {
                        int in_y = out_y + i - KERNEL_HALF;
                        int in_x = out_x + j - KERNEL_HALF;
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
                            result += input[c * height * width + in_y * width + in_x] * kernel[c * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j];
                    }
                }
            }
            result += bias;
            if (activation_num == ACTIVATION_RELU) {
                result = max(result, 0.0f);
            } else if (activation_num == ACTIVATION_SIGM) {
                result = 1 / (1 + exp(-result));
            }
            output[out_y * width + out_x] = result;
        }
    }
}

void Conv2d::load_weights(const char* weights_fname, const char* bias_fname) {
    int num_weights = out_ch * in_ch * k_size * k_size;
    {
        std::ifstream ifile(weights_fname, std::ios::binary);
        if (!ifile.is_open()) {
            std::cerr << "File with weights do not exist: " << weights_fname << std::endl;
        }
        ifile.read((char*)weights_host, num_weights * sizeof(float));
        ifile.close();
    }
    {
        std::ifstream ifile(bias_fname, std::ios::binary);
        if (!ifile.is_open()) {
            std::cerr << "File with biases do not exist: " << bias_fname << std::endl;
        }
        ifile.read((char*)bias_host, out_ch * sizeof(float));
        ifile.close();
    }

    cudaMemcpy(weights_device, weights_host,  num_weights * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(bias_device, bias_host,  out_ch * sizeof(float), cudaMemcpyHostToDevice);
}

void Conv2d::forward(const float* input_device, float* output_device, int height, int width, int activation_num) const {
    dim3 grid_dim((width + BLOCK_SZ_2D - 1) / BLOCK_SZ_2D, (height + BLOCK_SZ_2D - 1) / BLOCK_SZ_2D);
    dim3 block_dim(BLOCK_SZ_2D, BLOCK_SZ_2D);
    int one_filter_size = in_ch * k_size * k_size;
    int feature_map_size = height * width;
    for (int i = 0; i < out_ch; ++i) {
        conv2d<<<grid_dim, block_dim>>>(
            input_device, 
            output_device + i * feature_map_size, 
            weights_device + i * one_filter_size,
            bias_host[i], activation_num,
            height, width, in_ch
        );
    }
}

__global__ void conv2d_transpose(const float* input, float* output, const float* kernel, float bias, int activation_num, int height, int width, int channels) {
    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // loop replaces if statement, in most cases calculated once
    for (int out_y = y; out_y < height * 2; out_y += ny) {
        for (int out_x = x; out_x < width * 2; out_x += nx) {
            int k_x = out_x % KERNEL_TRANSPOSE;
            int k_y = out_y % KERNEL_TRANSPOSE;
            float result = 0.0f;
            for (int c = 0; c < channels; ++c) {
                result += input[c * width * height + int(out_y / 2) * width + int(out_x / 2)] * kernel[c * KERNEL_TRANSPOSE * KERNEL_TRANSPOSE + k_y * KERNEL_TRANSPOSE + k_x];
            }
            result += bias;
            if (activation_num == ACTIVATION_RELU) {
                result = max(result, 0.0f);
            } else if (activation_num == ACTIVATION_SIGM) {
                result = 1 / (1 + exp(-result));
            }
            output[out_y * width * 2 + out_x] = result;
        }
    }
}

void Conv2d::forward_transpose(const float* input_device, float* output_device, int height, int width, int activation_num) const {
    dim3 grid_dim((width * 2 + BLOCK_SZ_2D - 1) / BLOCK_SZ_2D, (height * 2 + BLOCK_SZ_2D - 1) / BLOCK_SZ_2D);
    dim3 block_dim(BLOCK_SZ_2D, BLOCK_SZ_2D);
    int one_filter_size = in_ch * k_size * k_size;
    int feature_map_size = height * width * 2 * 2;
    for (int i = 0; i < out_ch; ++i) {
        conv2d_transpose<<<grid_dim, block_dim>>>(
            input_device, 
            output_device + i * feature_map_size, 
            weights_device + i * one_filter_size,
            bias_host[i], activation_num,
            height, width, in_ch
        );
    }
}
