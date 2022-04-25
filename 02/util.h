#pragma once

#define KERNEL_SIZE 3
#define KERNEL_HALF 1
#define KERNEL_TRANSPOSE 2

#define BLOCK_SZ_1D 1024
#define BLOCK_SZ_2D 32
#define BLOCK_SZ_3D 8

#define INTRA_CH 32

#define CLIP(x, a, b) max(min((x), (b)), (a))

#include <chrono>
#include <iostream>

class Timer {
    std::string timer_name;
    std::chrono::steady_clock::time_point start_point;
    bool started = false;
public:
    std::chrono::microseconds elapsed_time;
    Timer(const std::string &timer_name) : timer_name(timer_name), elapsed_time(0) {}

    inline void start() {
        started = true;
        start_point = std::chrono::steady_clock::now();
    }
    inline void end() {
        std::chrono::steady_clock::time_point end_point = std::chrono::steady_clock::now();
        if (started) {
            elapsed_time += std::chrono::duration_cast<std::chrono::microseconds>(end_point - start_point);
        }
        started = false;
    }

    ~Timer() {
        std::cout << "Elapsed time on " << timer_name << ":\t" << elapsed_time.count() << "\t[microseconds]" << std::endl;
    }
};
