#pragma once

#define KERNEL_SIZE 3
#define KERNEL_HALF 1

#define BLOCK_SZ_1D 1024
#define BLOCK_SZ_2D 32
#define BLOCK_SZ_3D 8

#define INTRA_CH 32

#define CLIP(x, a, b) max(min((x), (b)), (a))
