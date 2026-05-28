#ifndef DCT2D_CUH
#define DCT2D_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include "DCTCommon.cuh"

class DCT2D {
public:
    DCT2D(int n1, int n2, bool ortho = true);
    ~DCT2D();

    DCT2D(const DCT2D&) = delete;
    DCT2D& operator=(const DCT2D&) = delete;

    void forward(const float* d_in, float* d_out);
    void backward(const float* d_in, float* d_out);

    void set_stream(cudaStream_t s);

private:
    int N1 = 0;
    int N2 = 0;
    int Nh = 0;

    bool ortho = true;
    cudaStream_t stream = nullptr;

    float* d_v = nullptr;
    cufftComplex* d_Vh = nullptr;

    float* d_c1 = nullptr;
    float* d_s1 = nullptr;
    float* d_c2 = nullptr;
    float* d_s2 = nullptr;

    float* d_fwd1 = nullptr;
    float* d_fwd2 = nullptr;

    float* d_inv1 = nullptr;
    float* d_inv2 = nullptr;

    cufftHandle plan_r2c = 0;
    cufftHandle plan_c2r = 0;
};

#endif // DCT2D_CUH