#pragma once

class Conv2d {
    int in_ch, out_ch;

    float* weights_host;
    float* weights_device;

public:
    Conv2d(int in_channels, int out_channels) : in_ch(in_channels), out_ch(out_channels) {
        int num_weights = out_ch * in_ch * KERNEL_SIZE * KERNEL_SIZE;
        cudaMalloc(&weights_device, num_weights * sizeof(float));
        weights_host = new float[num_weights];
    }
    
    ~Conv2d() {
        delete[] weights_host;
        cudaFree(weights_device);
    }

    void load_weights(const char* fname);
    void forward(const float* input_device, float* output_device, int height, int width) const;
    void forward_transpose(const float* input_device, float* output_device, int height, int width) const;
};
