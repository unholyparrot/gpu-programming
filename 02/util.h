#pragma once

#define KERNEL_SIZE 3
#define KERNEL_HALF 1

#define BLOCK_SIZE 32

#define INTRA_CH 32

#define CLIP(x, a, b) max(min((x), (b)), (a))
