#pragma once

class Conv2d {
    int in_ch, out_ch;
    int k_size;

    float* weights_host;
    float* weights_device;

    float* bias_host;
    // float* bias_device;

public:
    Conv2d(int in_channels, int out_channels, int kernel_size) : in_ch(in_channels), out_ch(out_channels), k_size(kernel_size) {
        int num_weights = out_ch * in_ch * k_size * k_size;

        weights_host = new float[num_weights];
        bias_host = new float[out_ch];
        
        cudaMalloc(&weights_device, num_weights * sizeof(float));
        // cudaMalloc(&bias_device, out_ch * sizeof(float));
    }
    
    ~Conv2d() {
        delete[] weights_host;
        delete[] bias_host;

        cudaFree(weights_device);
        // cudaFree(bias_device);
    }

    void load_weights(const char* weights_fname, const char* bias_fname);
    void forward(const float* input_device, float* output_device, int height, int width, int activation_num) const;
    void forward_optim(const float* input_device, float* output_device, int height, int width, int activation_num) const;
    void forward_transpose(const float* input_device, float* output_device, int height, int width, int activation_num) const;
};
