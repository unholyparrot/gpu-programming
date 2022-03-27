#include <cstdint>

__global__ void histogram_local_GPU_globalmem(const uint8_t *gray_img, int* all_hists, int height, int width);

__global__ void histogram_local_GPU(const uint8_t *gray_img, int* all_hists, int height, int width);

__global__ void histogram_final_GPU(const int *all_hists, int *hist, int num_hists);
