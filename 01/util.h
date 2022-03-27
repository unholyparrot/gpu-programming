#include <chrono>
#include <string>
#include <iostream>

#define RED         0
#define GREEN       1
#define BLUE        2
#define NUM_COLORS  3

#define Y_RED   0.2125f
#define Y_GREEN 0.7154f
#define Y_BLUE  0.0721f

#define Y_LEVELS 256
#define BLOCK_SZ 32


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
