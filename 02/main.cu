#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>


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