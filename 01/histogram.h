#include <cstdint>

__global__ void histogram_global_GPU(const uint8_t *gray_img, int* hist, int height, int width);

__global__ void histogram_local_globalmem_GPU(const uint8_t *gray_img, int* all_hists, int height, int width);

__global__ void histogram_local_sharedmem_GPU(const uint8_t *gray_img, int* all_hists, int height, int width);

__global__ void histogram_finalize_GPU(const int *all_hists, int *hist, int num_hists);
