#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#define RED     0
#define GREEN   1
#define BLUE    2
#define Y_RED   0.2125f
#define Y_GREEN 0.7154f
#define Y_BLUE  0.0721f

#define Y_LEVELS 256


__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

void rgb2gray_CPU(uint8_t* rgb_image, uint8_t* gray_image, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            *gray_image++ = Y_RED * rgb_image[RED] + Y_GREEN * rgb_image[GREEN] + Y_BLUE * rgb_image[BLUE] + 0.5;
            rgb_image += 3;
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

    // Allocate memory, initialize arrays
    uint8_t* gray_img = new uint8_t[img_h * img_w];
    int histogram[Y_LEVELS] = {};
    float scaling_coeff[Y_LEVELS] = {};

    // Start processing. CPU
    rgb2gray_CPU(rgb_img, gray_img, img_h, img_w);
#ifdef _DEBUG
    {
        int res = stbi_write_png("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_grayscale.png", img_w, img_h, 1, gray_img, 0);
        if (!res) {
            std::cout << stbi_failure_reason() << std::endl;
            return 1;
        }
    }
#endif

    histogram_CPU(gray_img, histogram, img_h, img_w);
#ifdef _DEBUG
    {
        std::ofstream out_f("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_hist.txt");
        for (int i = 0; i < Y_LEVELS; ++i) {
            out_f << histogram[i] << "\n";
        }
    }
#endif
    
    create_mapper(histogram, scaling_coeff, img_h * img_w);
#ifdef _DEBUG
    {
        std::ofstream out_f("C:/Users/kosto/Desktop/work/gpu_programming/misc_files/_debug_map.txt");
        for (int i = 0; i < Y_LEVELS; ++i) {
            out_f << scaling_coeff[i] << "\n";
        }
    }
#endif



    // Start processing. GPU

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
