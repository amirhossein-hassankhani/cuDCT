#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

constexpr float PI_F = 3.14159265358979323846f;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(cudaGetErrorString(err__));               \
        }                                                                      \
    } while (0)
#endif

#ifndef CHECK_CUFFT
#define CHECK_CUFFT(call)                                                       \
    do {                                                                        \
        cufftResult err__ = (call);                                              \
        if (err__ != CUFFT_SUCCESS) {                                            \
            throw std::runtime_error("cuFFT error");                            \
        }                                                                       \
    } while (0)
#endif

static inline int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

// ============================================================
// Index helpers
// ============================================================

__device__ __forceinline__ int idx2(int i1, int i2, int N2)
{
    return i1 * N2 + i2;
}

__device__ __forceinline__ int idx2h(int k1, int k2, int Nh)
{
    return k1 * Nh + k2;
}

__device__ __forceinline__ int idx3(int i1, int i2, int i3, int N2, int N3)
{
    return (i1 * N2 + i2) * N3 + i3;
}

__device__ __forceinline__ int idx3h(int k1, int k2, int k3, int N2, int Nh)
{
    return (k1 * N2 + k2) * Nh + k3;
}

// ============================================================
// Makhoul mapping
// ============================================================

__device__ __forceinline__ int makhoul_map(int i, int N)
{
    int half = (N + 1) >> 1;
    return (i < half) ? (2 * i) : (2 * (N - 1 - i) + 1);
}

__device__ __forceinline__ int makhoul_inv_map(int j, int N)
{
    return ((j & 1) == 0)
        ? (j >> 1)
        : (N - 1 - ((j - 1) >> 1));
}

__device__ __forceinline__ int flip0(int k, int N)
{
    return (k == 0) ? 0 : (N - k);
}

__device__ __forceinline__ int mod_neg(int k, int N)
{
    return (k == 0) ? 0 : (N - k);
}

// ============================================================
// Complex helpers
// ============================================================

__device__ __forceinline__ cufftComplex cadd(cufftComplex a, cufftComplex b)
{
    return cufftComplex{a.x + b.x, a.y + b.y};
}

__device__ __forceinline__ cufftComplex mul_phase_minus(
    cufftComplex z,
    float c,
    float s)
{
    // z * exp(-j theta)
    return cufftComplex{
        c * z.x + s * z.y,
        c * z.y - s * z.x
    };
}

__device__ __forceinline__ cufftComplex mul_phase_plus(
    cufftComplex z,
    float c,
    float s)
{
    // z * exp(+j theta)
    return cufftComplex{
        c * z.x - s * z.y,
        s * z.x + c * z.y
    };
}
