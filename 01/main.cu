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

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

void rgb2gray_CPU(const uint8_t* rgb_image, uint8_t* gray_image, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            *gray_image++ = Y_RED * rgb_image[RED] + Y_GREEN * rgb_image[GREEN] + Y_BLUE * rgb_image[BLUE];
            rgb_image += 3;
        }
    }
}

__global__ void rgb2gray_GPU(const uint8_t* rgb_image, uint8_t* gray_image, int h, int w) {
    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    for (int i = y; i < h; i += ny) {
        for (int j = x; j < w; j += nx) {
            const uint8_t* in_pixel = rgb_image + NUM_COLORS * (i * w + j);
            uint8_t* out_pixel = gray_image + i * w + j;

            *out_pixel = Y_RED * in_pixel[RED] + Y_GREEN * in_pixel[GREEN] + Y_BLUE * in_pixel[BLUE];
        }
    }
}

void histogram_CPU(uint8_t* gray_image, int* hist, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            ++hist[*gray_image++];
        }
    }
}


__global__ void histogram_local_GPU(const uint8_t *gray_img, int* all_hist, int h, int w) {
    int t = threadIdx.y * blockDim.x + threadIdx.x; // thread linear idx in block
    int num_threads = blockDim.x * blockDim.y;      // thread total count in block
    
    __shared__ int local_hist[Y_LEVELS];
    for (int i = t; i < Y_LEVELS; i += num_threads)
        local_hist[i] = 0;
    __syncthreads();

    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // number of processed pixels
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    for (int i = y; i < h; i += ny)
        for (int j = x; j < w; j += nx)
            atomicAdd(local_hist + gray_img[i * w + j], 1);
    __syncthreads();

    all_hist += (blockIdx.y * gridDim.x + blockIdx.x) * Y_LEVELS;
    for (int i = t; i < Y_LEVELS; i += num_threads)
        all_hist[i] = local_hist[i];
}

__global__ void histogram_final_GPU(const int *all_hist, int *hist, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; // thread global idx
    if (t < Y_LEVELS) {
        int total = 0;
        for (int j = 0; j < n; j++) 
            total += all_hist[j * Y_LEVELS + t];
        hist[t] = total;
    }
}

void create_mapper(int* hist, float* scaling_coeff, int pixel_count) {
    int cumsum[Y_LEVELS] = {};
    cumsum[0] = hist[0];
    scaling_coeff[0] = 0;
    for (int i = 1; i < Y_LEVELS; ++i) {
        cumsum[i] = cumsum[i-1] + hist[i];
        // mapper[i] = (Y_LEVELS * cumsum[i] + pixel_count - 1) / pixel_count - 1;
        scaling_coeff[i] = static_cast<float>((Y_LEVELS * cumsum[i] + pixel_count - 1) / pixel_count - 1) / i;
    }
}

void autocontrast_CPU(uint8_t* rgb_src, uint8_t* rgb_dst, uint8_t* gray_img, float* scaling_coef, int h, int w, int c) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float coef = scaling_coef[*gray_img++];
            for (int k = 0; k < c; ++k) { // RGB or Y
                *rgb_dst++ = std::min(*rgb_src++ * coef, 255.0f);
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
    dim3 grid_dim((img_w + BLOCK_SZ - 1) / BLOCK_SZ, (img_h + BLOCK_SZ - 1) / BLOCK_SZ);
    dim3 block_dim(BLOCK_SZ, BLOCK_SZ);

    cudaMalloc(&rgb_img_device, img_h * img_w * img_c);
    cudaMalloc(&gray_img_device, img_h * img_w);
    cudaMalloc(&res_img_device, img_h * img_w * img_c);
    cudaMalloc(&all_hist_device, Y_LEVELS * grid_dim.x * grid_dim.y * sizeof(int));
    cudaMalloc(&histogram_device, Y_LEVELS * sizeof(int));    

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

    return 0;




    int N = 1<<20;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
