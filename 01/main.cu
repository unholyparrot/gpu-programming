#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#define RED         0
#define GREEN       1
#define BLUE        2
#define NUM_COLORS  3

#define Y_RED   0.2125f
#define Y_GREEN 0.7154f
#define Y_BLUE  0.0721f

#define Y_LEVELS 256
#define BLOCK_SZ 32

void rgb2gray_CPU(const uint8_t* rgb_image, uint8_t* gray_image, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            *gray_image++ = Y_RED * rgb_image[RED] + Y_GREEN * rgb_image[GREEN] + Y_BLUE * rgb_image[BLUE];
            rgb_image += NUM_COLORS;
        }
    }
}

__global__ void rgb2gray_GPU(const uint8_t* rgb_image, uint8_t* gray_image, int height, int width) {
    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    for (int i = y; i < height; i += ny) {
        for (int j = x; j < width; j += nx) {
            int linear_idx = i * width + j;
            const uint8_t* in_pixel = rgb_image + NUM_COLORS * linear_idx;
            uint8_t* out_pixel = gray_image + linear_idx;

            *out_pixel = Y_RED * in_pixel[RED] + Y_GREEN * in_pixel[GREEN] + Y_BLUE * in_pixel[BLUE];
        }
    }
}

void histogram_CPU(const uint8_t* gray_img, int* hist, int height, int width) {
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            ++hist[*gray_img++];
}

__global__ void histogram_local_GPU(const uint8_t *gray_img, int* all_hists, int height, int width) {
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

__global__ void histogram_final_GPU(const int *all_hists, int *hist, int num_hists) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; // thread global idx
    if (t < Y_LEVELS) {
        int total = 0;
        for (int i = 0; i < num_hists; i++) 
            total += all_hists[i * Y_LEVELS + t];
        hist[t] = total;
    }
}

void create_mapper(const int* hist, float* scaling_coeff, int pixel_count) {
    int cumsum[Y_LEVELS] = {};
    cumsum[0] = hist[0];
    scaling_coeff[0] = 0;
    for (int i = 1; i < Y_LEVELS; ++i) {
        cumsum[i] = cumsum[i-1] + hist[i];
        // mapper[i] = (Y_LEVELS * cumsum[i] + pixel_count - 1) / pixel_count - 1;
        scaling_coeff[i] = static_cast<float>((Y_LEVELS * cumsum[i] + pixel_count - 1) / pixel_count - 1) / i;
    }
}

void autocontrast_CPU(
    const uint8_t* rgb_src, uint8_t* rgb_dst,
    const uint8_t* gray_img, const float* scaling_coef,
    int height, int width, int channels)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float coef = scaling_coef[*gray_img++];
            for (int k = 0; k < channels; ++k) { // RGB or Y
                *rgb_dst++ = std::min(*rgb_src++ * coef, 255.0f);
            }
        }
    }
}

__global__ void autocontrast_GPU(
    const uint8_t* rgb_src, uint8_t* rgb_dst,
    const uint8_t* gray_img, const float* scaling_coef,
    int height, int width, int channels)
{
    // coordinates of the first pixel to process
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels by step
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    for (int i = y; i < height; i += ny) {
        for (int j = x; j < width; j += nx) {
            int linear_idx = i * width + j;
            const uint8_t* in_pixel = rgb_src + NUM_COLORS * linear_idx;
            uint8_t* out_pixel = rgb_dst + NUM_COLORS * linear_idx;
            float coef = scaling_coef[gray_img[linear_idx]];

            for (int k = 0; k < channels; ++k) { // RGB or Y
                *out_pixel++ = min(*in_pixel++ * coef, 255.0f);
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: <input_image> <output_image>" << std::endl;
        return 0;
    }
    /// Load image
    int img_h, img_w, img_c;
    uint8_t* rgb_img = stbi_load(argv[1], &img_w, &img_h, &img_c, 0);
    if (!rgb_img) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Image loaded successfully. Shape: (" << img_h << ", " << img_w << ", " << img_c << ")" << std::endl;

    // Start processing. CPU
    // Allocate memory, initialize arrays
    uint8_t* gray_img = new uint8_t[img_h * img_w];
    uint8_t* res_img = new uint8_t[img_h * img_w * img_c];
    int histogram[Y_LEVELS] = {};
    float scaling_coeff[Y_LEVELS] = {};

    rgb2gray_CPU(rgb_img, gray_img, img_h, img_w);
#ifdef _DEBUG
    {
        int res = stbi_write_png("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_grayscale_CPU.png", img_w, img_h, 1, gray_img, 0);
        if (!res) {
            std::cout << stbi_failure_reason() << std::endl;
            return 1;
        }
    }
#endif

    histogram_CPU(gray_img, histogram, img_h, img_w);
#ifdef _DEBUG
    {
        std::ofstream out_f("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_hist_CPU.txt");
        for (int i = 0; i < Y_LEVELS; ++i) {
            out_f << histogram[i] << "\n";
        }
    }
#endif
    
    create_mapper(histogram, scaling_coeff, img_h * img_w);
#ifdef _DEBUG
    {
        std::ofstream out_f("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_map_CPU.txt");
        for (int i = 0; i < Y_LEVELS; ++i) {
            out_f << scaling_coeff[i] << " " << scaling_coeff[i] * i << "\n";
        }
    }
#endif
    
#ifdef _DEBUG
    {
        autocontrast_CPU(gray_img, res_img, gray_img, scaling_coeff, img_h, img_w, 1);
        int res = stbi_write_png("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_result_gray_CPU.png", img_w, img_h, 1, res_img, 0);
        if (!res) {
            std::cout << stbi_failure_reason() << std::endl;
            return 1;
        }
    }
#endif

    autocontrast_CPU(rgb_img, res_img, gray_img, scaling_coeff, img_h, img_w, img_c);
#ifdef _DEBUG
    {
        int res = stbi_write_png("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_result_CPU.png", img_w, img_h, img_c, res_img, 0);
        if (!res) {
            std::cout << stbi_failure_reason() << std::endl;
            return 1;
        }
    }
#endif

    // Save image
    int res = stbi_write_png(argv[2], img_w, img_h, img_c, res_img, 0);
    if (!res) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }

    // Start processing. GPU
    uint8_t *rgb_img_device, *gray_img_device, *res_img_device;
    int *all_hist_device, *histogram_device;
    float *scaling_coeff_device;
    dim3 grid_dim((img_w + BLOCK_SZ - 1) / BLOCK_SZ, (img_h + BLOCK_SZ - 1) / BLOCK_SZ);
    dim3 block_dim(BLOCK_SZ, BLOCK_SZ);

    cudaMalloc(&rgb_img_device, img_h * img_w * img_c);
    cudaMalloc(&gray_img_device, img_h * img_w);
    cudaMalloc(&res_img_device, img_h * img_w * img_c);
    cudaMalloc(&all_hist_device, Y_LEVELS * grid_dim.x * grid_dim.y * sizeof(int));
    cudaMalloc(&histogram_device, Y_LEVELS * sizeof(int));
    cudaMalloc(&scaling_coeff_device, Y_LEVELS * sizeof(float));

    cudaMemcpy(rgb_img_device, rgb_img, img_h * img_w * img_c, cudaMemcpyHostToDevice);
    rgb2gray_GPU<<<grid_dim, block_dim>>>(rgb_img_device, gray_img_device, img_h, img_w);
#ifdef _DEBUG
    {
        cudaMemcpy(gray_img, gray_img_device, img_h * img_w, cudaMemcpyDeviceToHost);
        int res = stbi_write_png("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_grayscale_GPU.png", img_w, img_h, 1, gray_img, 0);
        if (!res) {
            std::cout << stbi_failure_reason() << std::endl;
            return 1;
        }
    }
#endif

    histogram_local_GPU<<<grid_dim, block_dim>>>(gray_img_device, all_hist_device, img_h, img_w);
    histogram_final_GPU<<<1, Y_LEVELS>>>(all_hist_device, histogram_device, grid_dim.x * grid_dim.y);
#ifdef _DEBUG
    {
        int debug_hist[Y_LEVELS] = {};
        cudaMemcpy(debug_hist, histogram_device, Y_LEVELS * sizeof(int), cudaMemcpyDeviceToHost);
        std::ofstream out_f("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_hist_GPU.txt");
        for (int i = 0; i < Y_LEVELS; ++i) {
            out_f << debug_hist[i] << "\n";
        }
    }
#endif

    cudaMemcpy(histogram, histogram_device, Y_LEVELS * sizeof(int), cudaMemcpyDeviceToHost);
    create_mapper(histogram, scaling_coeff, img_h * img_w);
    cudaMemcpy(scaling_coeff_device, scaling_coeff, Y_LEVELS * sizeof(float), cudaMemcpyHostToDevice);

    autocontrast_GPU<<<grid_dim, block_dim>>>(rgb_img_device, res_img_device, gray_img_device, scaling_coeff_device, img_h, img_w, img_c);
    cudaMemcpy(res_img, res_img_device, img_h * img_w * img_c, cudaMemcpyDeviceToHost);
#ifdef _DEBUG
    {
        int res = stbi_write_png("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_result_GPU.png", img_w, img_h, img_c, res_img, 0);
        if (!res) {
            std::cout << stbi_failure_reason() << std::endl;
            return 1;
        }
    }
#endif

    // Save image
    std::string out_fname(argv[2]);
    out_fname += "GPU.png";
    res = stbi_write_png(out_fname.c_str(), img_w, img_h, img_c, res_img, 0);
    if (!res) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }



    return 0;
}
