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

// #define Y_RED   0.299
// #define Y_GREEN 0.587
// #define Y_BLUE  0.114

#define HISTOGRAM_BINS 256



__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

void rgb2gray_CPU(uint8_t* rgb_image, uint8_t* gray_image, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            *gray_image++ = Y_RED * rgb_image[RED] + Y_GREEN * rgb_image[GREEN] + Y_BLUE * rgb_image[BLUE];
            rgb_image += 3;
        }
    }
}

void float2uint(float* src_img, uint8_t* dst_img, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            *dst_img++ = static_cast<uint8_t>(*src_img++ + 0.5);
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: <input_image> <output_image>" << std::endl;
        return 0;
    }
    int img_h, img_w, img_c;
    uint8_t* rgb_img = stbi_load(argv[1], &img_w, &img_h, &img_c, 0);
    if (!rgb_img) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Image loaded successfully. Shape: (" << img_h << ", " << img_w << ", " << img_c << ")" << std::endl;

    int plane_size = img_h * img_w;
    uint8_t* gray_img = new uint8_t[plane_size];
    rgb2gray_CPU(rgb_img, gray_img, img_h, img_w);

    int res = stbi_write_png(argv[2], img_w, img_h, 1, gray_img, 0);
    if (!res) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }

    int histogram[HISTOGRAM_BINS] = {};
    histogram_CPU(gray_img, histogram, img_h, img_w);

    std::ofstream out_f("C:/Users/kosto/Desktop/work/gpu_programming/hist.txt");
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        out_f << histogram[i] << "\n";
    }
    out_f.close();

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
