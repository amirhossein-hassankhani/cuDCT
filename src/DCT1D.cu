#include "DCT1D.cuh"
#include <vector>
#include <cmath>

// ============================================================
// Reorder / unreorder
// ============================================================

__global__ void k_reorder_to_fft_in_1d(
    const float* __restrict__ x,
    float* __restrict__ v,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int j = makhoul_map(i, N);

    v[i] = x[j];
}

__global__ void k_unreorder_from_fft_out_scaled_1d(
    const float* __restrict__ v,
    float* __restrict__ x,
    int N,
    float scale)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    int i = makhoul_inv_map(j, N);

    x[j] = scale * v[i];
}

// ============================================================
// Forward merge: R2C half-spectrum -> DCT coefficients
// ============================================================

template <bool ORTHO>
__global__ void k_dct1_forward_merge(
    const cufftComplex* __restrict__ Vh,
    float* __restrict__ C,
    const float* __restrict__ c,
    const float* __restrict__ s,
    const float* __restrict__ fwd,
    int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    int Nh = N / 2 + 1;

    cufftComplex V;

    if (k < Nh)
    {
        V = Vh[k];
    }
    else
    {
        int kb = N - k;
        V = Vh[kb];
        V.y = -V.y;
    }

    cufftComplex R = mul_phase_minus(V, c[k], s[k]);

    float val = R.x;

    if constexpr (ORTHO)
        val *= fwd[k];

    C[k] = val;
}

// ============================================================
// Backward merge: DCT coefficients -> R2C half-spectrum
// ============================================================
template <bool ORTHO>
__global__ void k_idct1_build_Vh(
    const float* __restrict__ C,
    cufftComplex* __restrict__ Vh,
    const float* __restrict__ c,
    const float* __restrict__ s,
    const float* __restrict__ inv,
    int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    int Nh = N / 2 + 1;
    if (k >= Nh) return;

    int kb = flip0(k, N);

    float re = C[k];
    float im = (k == 0) ? 0.0f : -C[kb];

    if constexpr (ORTHO)
    {
        re *= inv[k];
        if (k != 0)
            im *= inv[kb];
    }

    cufftComplex V{re, im};

    V = mul_phase_plus(V, c[k], s[k]);

    Vh[k] = V;
}

// ============================================================
// DCT1D class
// ============================================================

DCT1D::DCT1D(int n, bool use_ortho)
    : N(n), Nh(n / 2 + 1), ortho(use_ortho)
{
    if (N <= 0)
        throw std::runtime_error("Invalid DCT1D size");

    stream = 0;

    size_t real_bytes = size_t(N) * sizeof(float);
    size_t half_bytes = size_t(Nh) * sizeof(cufftComplex);

    CHECK_CUDA(cudaMalloc(&d_v, real_bytes));
    CHECK_CUDA(cudaMalloc(&d_Vh, half_bytes));

    CHECK_CUDA(cudaMalloc(&d_c, size_t(N) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s, size_t(N) * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_fwd, size_t(N) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inv, size_t(N) * sizeof(float)));

    std::vector<float> h_c(N);
    std::vector<float> h_s(N);

    for (int k = 0; k < N; ++k)
    {
        float theta = PI_F * float(k) / (2.0f * float(N));
        h_c[k] = std::cos(theta);
        h_s[k] = std::sin(theta);
    }

    CHECK_CUDA(cudaMemcpy(
        d_c, h_c.data(),
        size_t(N) * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_s, h_s.data(),
        size_t(N) * sizeof(float),
        cudaMemcpyHostToDevice));

    std::vector<float> h_fwd(N);
    std::vector<float> h_inv(N);

    for (int k = 0; k < N; ++k)
    {
        float a = (k == 0)
            ? 1.0f / std::sqrt(float(N))
            : std::sqrt(2.0f / float(N));

        // 1D raw forward is Re(e^{-j theta} V).
        // To match orthonormal DCT-II:
        //
        // C_ortho[k] = a[k] * C_raw[k]
        //
        h_fwd[k] = a;

        // Inverse undo:
        //
        // C_raw[k] = C_ortho[k] / a[k]
        //
        h_inv[k] = 1.0f / a;
    }

    CHECK_CUDA(cudaMemcpy(
        d_fwd, h_fwd.data(),
        size_t(N) * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_inv, h_inv.data(),
        size_t(N) * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUFFT(cufftPlan1d(&plan_r2c, N, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&plan_c2r, N, CUFFT_C2R, 1));

    CHECK_CUFFT(cufftSetStream(plan_r2c, stream));
    CHECK_CUFFT(cufftSetStream(plan_c2r, stream));
}

DCT1D::~DCT1D()
{
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    cudaFree(d_v);
    cudaFree(d_Vh);

    cudaFree(d_c);
    cudaFree(d_s);

    cudaFree(d_fwd);
    cudaFree(d_inv);
}

void DCT1D::set_stream(cudaStream_t s)
{
    stream = s;

    cufftSetStream(plan_r2c, stream);
    cufftSetStream(plan_c2r, stream);
}

void DCT1D::forward(const float* d_in, float* d_out)
{
    int block = 512;
    int grid = div_up(N, block);

    k_reorder_to_fft_in_1d<<<grid, block, 0, stream>>>(
        d_in, d_v, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecR2C(
        plan_r2c,
        reinterpret_cast<cufftReal*>(d_v),
        reinterpret_cast<cufftComplex*>(d_Vh)));

    if (ortho)
    {
        k_dct1_forward_merge<true><<<grid, block, 0, stream>>>(
            d_Vh, d_out,
            d_c, d_s,
            d_fwd,
            N);
    }
    else
    {
        k_dct1_forward_merge<false><<<grid, block, 0, stream>>>(
            d_Vh, d_out,
            d_c, d_s,
            nullptr,
            N);
    }

    CHECK_CUDA(cudaGetLastError());
}

void DCT1D::backward(const float* d_in, float* d_out)
{
    int block = 512;
    int grid_full = div_up(N, block);
    int grid_half = div_up(Nh, block);

    if (ortho)
    {
        k_idct1_build_Vh<true><<<grid_half, block, 0, stream>>>(
            d_in, d_Vh,
            d_c, d_s,
            d_inv,
            N);
    }
    else
    {
        k_idct1_build_Vh<false><<<grid_half, block, 0, stream>>>(
            d_in, d_Vh,
            d_c, d_s,
            nullptr,
            N);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecC2R(
        plan_c2r,
        reinterpret_cast<cufftComplex*>(d_Vh),
        reinterpret_cast<cufftReal*>(d_v)));

    float scale = 1.0f / float(N);

    k_unreorder_from_fft_out_scaled_1d<<<grid_full, block, 0, stream>>>(
        d_v, d_out, N, scale);
    CHECK_CUDA(cudaGetLastError());
}