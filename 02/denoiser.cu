#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <sstream>

#include "util.h"
#include "convolution.h"
#include "modules.h"



class Model {
    Conv2d conv0, conv3, tconv6, tconv8, conv10;
    ReLU relu;
    Sigmoid sigmoid;
    MaxPool2d maxpool;

public:
    Model() 
    : conv0(1, INTRA_CH, KERNEL_SIZE)
    , conv3(INTRA_CH, INTRA_CH, KERNEL_SIZE)
    , tconv6(INTRA_CH, INTRA_CH, KERNEL_TRANSPOSE)
    , tconv8(INTRA_CH, INTRA_CH, KERNEL_TRANSPOSE)
    , conv10(INTRA_CH, 1, KERNEL_SIZE)
    {
        conv0.load_weights("model/0.weight.bin", "model/0.bias.bin");
        conv3.load_weights("model/3.weight.bin", "model/3.bias.bin");
        tconv6.load_weights("model/6.weight.bin", "model/6.bias.bin");
        tconv8.load_weights("model/8.weight.bin", "model/8.bias.bin");
        conv10.load_weights("model/10.weight.bin", "model/10.bias.bin");
    }

    // input, output -- device memory of the same size HxWx1
    void forward(const float* input, float* output, int height, int width) {
        float *tmp1, *tmp2;
        cudaMalloc(&tmp1, height * width * INTRA_CH * sizeof(float));
        cudaMalloc(&tmp2, height * width * INTRA_CH * sizeof(float));

        conv0.forward(input, tmp1, height, width);
        relu.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);
        maxpool.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);
        height = (height + 1) / 2; width = (width + 1) / 2;

        conv3.forward(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        relu.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);
        maxpool.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);
        height = (height + 1) / 2; width = (width + 1) / 2;

        tconv6.forward_transpose(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        height *= 2; width *= 2;
        relu.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);

        tconv8.forward_transpose(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        height *= 2; width *= 2;
        relu.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);

        conv10.forward(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        sigmoid.forward(tmp1, output, height, width, 1);

        cudaFree(tmp1);
        cudaFree(tmp2);
    }

};

__global__ void img_byte2float(const uint8_t* input, float* output, int num_pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = blockDim.x * gridDim.x;
    for (int i = x; i < num_pixels; i+= nx) {
        output[i] = input[i] / 255.0f;
    }
}

__global__ void img_float2byte(const float* input, uint8_t* output, int num_pixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = blockDim.x * gridDim.x;
    for (int i = x; i < num_pixels; i+= nx) {
        output[i] = input[i] * 255.0f;
    }
}


void save_image(const char* filename, const uint8_t* img, int height, int width, int channels) {
    int res = stbi_write_png(filename, width, height, channels, img, 0);
    if (!res) {
        std::cout << stbi_failure_reason() << std::endl;
        exit(1);
    }
}


int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "-h")) {
        std::cout << "Usage: ./denoiser <input_image> [-benchmark X]" << std::endl;
        return 0;
    }
    std::string in_fname(argv[1]);
    std::string out_fname_gpu = in_fname.substr(0, in_fname.size() - 4) + "_denoised.png";
    int num_runs = 1;
    bool benchmark = (argc > 2 && (!strcmp(argv[2], "-b") || !strcmp(argv[2], "--benchmark")));
    if (benchmark) {
        if (argc > 3) {
            std::stringstream ss(argv[3]);
            ss >> num_runs;
        } else {
            num_runs = 10;
        }
        std::cout << "Starting benchmark: " << num_runs << " runs." << std::endl;
    }

    /// Load image
    int img_h, img_w, img_c;
    uint8_t* img = stbi_load(in_fname.c_str(), &img_w, &img_h, &img_c, 0);
    if (!img) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Image loaded successfully. Shape: (" << img_h << ", " << img_w << ", " << img_c << ")" << std::endl;
    int num_pixels = img_h * img_w * img_c;

    /// Allocate
    uint8_t *img_device;
    float *input_img_device, *output_img_device;
    cudaMalloc(&img_device, num_pixels * sizeof(uint8_t));
    cudaMalloc(&input_img_device, num_pixels * sizeof(float));
    cudaMalloc(&output_img_device, num_pixels * sizeof(float));
    Model model;

    /// Process
    cudaMemcpy(img_device, img, num_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice);
    img_byte2float<<<(num_pixels + BLOCK_SZ_1D - 1) / BLOCK_SZ_1D, BLOCK_SZ_1D>>>(img_device, input_img_device, num_pixels);
    cudaDeviceSynchronize();
    for (int i = 0; i < num_runs; ++i) {
        model.forward(input_img_device, output_img_device, img_h, img_w);
        cudaDeviceSynchronize();
    }
    img_float2byte<<<(num_pixels + BLOCK_SZ_1D - 1) / BLOCK_SZ_1D, BLOCK_SZ_1D>>>(output_img_device, img_device, num_pixels);
    cudaMemcpy(img, img_device, num_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    /// Save output
    save_image(out_fname_gpu.c_str(), img, img_h, img_w, img_c);
    return 0;
}