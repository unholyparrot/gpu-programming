#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>

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
        conv0.load_weights("model/weights_0.bin", "model/bias_0.bin");
        conv3.load_weights("model/weights_3.bin", "model/bias_3.bin");
        tconv6.load_weights("model/weights_6.bin", "model/bias_6.bin");
        tconv8.load_weights("model/weights_8.bin", "model/bias_8.bin");
        conv10.load_weights("model/weights_10.bin", "model/bias_10.bin");
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

        tconv6.forward(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        height *= 2; width *= 2;
        relu.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);

        tconv8.forward(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        height *= 2; width *= 2;
        relu.forward(tmp1, tmp2, height, width, INTRA_CH); std::swap(tmp1, tmp2);

        conv10.forward(tmp1, tmp2, height, width); std::swap(tmp1, tmp2);
        sigmoid.forward(tmp1, output, height, width, 1);

        cudaFree(tmp1);
        cudaFree(tmp2);
    }

};


void save_image(const char* filename, const uint8_t* img, int height, int width, int channels) {
    int res = stbi_write_png(filename, width, height, channels, img, 0);
    if (!res) {
        std::cout << stbi_failure_reason() << std::endl;
        exit(1);
    }
}


int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "-h")) {
        std::cout << "Usage: BABABA ./main <input_image> [-b --benchmark]" << std::endl;
        return 0;
    }
    std::string in_fname(argv[1]);
    std::string out_fname_gpu("out_gpu.png");
    bool benchmark = (argc > 2 && (!strcmp(argv[2], "-b") || !strcmp(argv[2], "--benchmark")));

    /// Load image
    int img_h, img_w, img_c;
    uint8_t* rgb_img = stbi_load(in_fname.c_str(), &img_w, &img_h, &img_c, 0);
    if (!rgb_img) {
        std::cout << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Image loaded successfully. Shape: (" << img_h << ", " << img_w << ", " << img_c << ")" << std::endl;
    uint8_t* res_img = new uint8_t[img_h * img_w * img_c];

    /// Process

    save_image(out_fname_gpu.c_str(), res_img, img_h, img_w, img_c);

    return 0;
}