#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include "DCTCommon.cuh"

class DCT1D {
public:
    DCT1D(int n, bool ortho = true);
    ~DCT1D();

    DCT1D(const DCT1D&) = delete;
    DCT1D& operator=(const DCT1D&) = delete;

    void forward(const float* d_in, float* d_out);
    void backward(const float* d_in, float* d_out);

    void set_stream(cudaStream_t s);

private:
    int N = 0;
    int Nh = 0;

    bool ortho = true;
    cudaStream_t stream = nullptr;

    float* d_v = nullptr;
    cufftComplex* d_Vh = nullptr;

    float* d_c = nullptr;
    float* d_s = nullptr;

    float* d_fwd = nullptr;
    float* d_inv = nullptr;

    cufftHandle plan_r2c = 0;
    cufftHandle plan_c2r = 0;
};