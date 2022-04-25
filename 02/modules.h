#include <iostream>
#include <fstream>

#include "util.h"

class ReLU {
public:
    void forward(const float* input_device, float* output_device, int height, int width, int channels) const;
};

class MaxPool2d {
public:
    void forward(const float* input_device, float* output_device, int height, int width, int channels) const;
};

class Sigmoid {
public:
    void forward(const float* input_device, float* output_device, int height, int width, int channels) const;
};