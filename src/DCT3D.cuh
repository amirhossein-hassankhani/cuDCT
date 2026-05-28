#ifndef DCT3D_CUH
#define DCT3D_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include "DCTCommon.cuh"
    
class DCT3D {
public:
    DCT3D(int n1, int n2, int n3, bool ortho);
    ~DCT3D();

    void forward(const float* d_in, float* d_out);
    void backward(const float* d_in, float* d_out);

    void set_stream(cudaStream_t s) {
        stream = s;
        cufftSetStream(plan_r2c, stream);
        cufftSetStream(plan_c2r, stream);
    }

private:
    int N1, N2, N3, Nh;
    bool ortho;
    cudaStream_t stream;

    float* d_v = nullptr;
    cufftComplex* d_Vh = nullptr;

    float *d_c1 = nullptr, *d_s1 = nullptr;
    float *d_c2 = nullptr, *d_s2 = nullptr;
    float *d_c3 = nullptr, *d_s3 = nullptr;

    float *d_fwd1 = nullptr, *d_fwd2 = nullptr, *d_fwd3 = nullptr;
    float *d_inv1 = nullptr, *d_inv2 = nullptr, *d_inv3 = nullptr;

    cufftHandle plan_r2c;
    cufftHandle plan_c2r;
};

#endif // DCT3D_CUH