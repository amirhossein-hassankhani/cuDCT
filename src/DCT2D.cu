#include "DCT2D.cuh"
#include <vector>
#include <cmath>

// ============================================================
// Reorder / unreorder
// ============================================================

__global__ void k_reorder_to_fft_in_2d(
    const float* __restrict__ x,
    float* __restrict__ v,
    int N1, int N2)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i1 >= N1 || i2 >= N2) return;

    int j1 = makhoul_map(i1, N1);
    int j2 = makhoul_map(i2, N2);

    v[idx2(i1, i2, N2)] = x[idx2(j1, j2, N2)];
}

__global__ void k_unreorder_from_fft_out_scaled_2d(
    const float* __restrict__ v,
    float* __restrict__ x,
    int N1, int N2,
    float scale)
{
    int j2 = blockIdx.x * blockDim.x + threadIdx.x;
    int j1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (j1 >= N1 || j2 >= N2) return;

    int i1 = makhoul_inv_map(j1, N1);
    int i2 = makhoul_inv_map(j2, N2);

    x[idx2(j1, j2, N2)] = scale * v[idx2(i1, i2, N2)];
}

// ============================================================
// Forward merge: R2C half-spectrum -> 2D DCT coefficients
// ============================================================

template <bool ORTHO>
__global__ void k_dct2_forward_merge(
    const cufftComplex* __restrict__ Vh,
    float* __restrict__ C,
    const float* __restrict__ c1,
    const float* __restrict__ s1,
    const float* __restrict__ c2,
    const float* __restrict__ s2,
    const float* __restrict__ fwd1,
    const float* __restrict__ fwd2,
    int N1, int N2)
{
    int k2 = blockIdx.x * blockDim.x + threadIdx.x;
    int k1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (k1 >= N1 || k2 >= N2) return;

    int Nh = N2 / 2 + 1;

    auto getV = [&](int a1, int a2) -> cufftComplex
    {
        if (a2 < Nh)
        {
            return Vh[idx2h(a1, a2, Nh)];
        }

        int b2 = N2 - a2;
        int b1 = mod_neg(a1, N1);

        cufftComplex t = Vh[idx2h(b1, b2, Nh)];
        t.y = -t.y;
        return t;
    };

    int k2b = flip0(k2, N2);

    cufftComplex V0  = getV(k1, k2);
    cufftComplex Vf2 = getV(k1, k2b);

    float ca = c1[k1], sa = s1[k1];
    float cb = c2[k2], sb = s2[k2];


    cufftComplex Q = cadd(
        mul_phase_minus(V0,  cb, sb),
        mul_phase_plus (Vf2, cb, sb)
    );

    cufftComplex R = mul_phase_minus(Q, ca, sa);

    float val = 2.0f * R.x;

    if constexpr (ORTHO)
        val *= fwd1[k1] * fwd2[k2];

    C[idx2(k1, k2, N2)] = val;
}

// ============================================================
// Backward merge: DCT coefficients -> R2C half-spectrum
// ============================================================

template <bool ORTHO>
__global__ void k_idct2_build_Vh(
    const float* __restrict__ C,
    cufftComplex* __restrict__ Vh,
    const float* __restrict__ c1,
    const float* __restrict__ s1,
    const float* __restrict__ c2,
    const float* __restrict__ s2,
    const float* __restrict__ inv1,
    const float* __restrict__ inv2,
    int N1, int N2)
{
    int k2 = blockIdx.x * blockDim.x + threadIdx.x;
    int k1 = blockIdx.y * blockDim.y + threadIdx.y;

    int Nh = N2 / 2 + 1;

    if (k1 >= N1 || k2 >= Nh) return;

    int k2b = (k2 == 0) ? 0 : (N2 - k2);

    auto load_C = [&](int i1, int i2) -> float
    {
        float v = C[idx2(i1, i2, N2)];

        if constexpr (ORTHO)
            v *= inv1[i1] * inv2[i2];

        return v;
    };

    // Chat(k1) = C(k1,k2) - j C(k1,N2-k2)
    auto load_re = [&](int i1) -> float
    {
        return load_C(i1, k2);
    };

    auto load_im = [&](int i1) -> float
    {
        return (k2 == 0) ? 0.0f : -load_C(i1, k2b);
    };

    float Xr = load_re(k1);
    float Xi = load_im(k1);

    if (k1 > 0)
    {
        int k1b = N1 - k1;


        Xr += load_im(k1b);
        Xi -= load_re(k1b);
    }

    cufftComplex V{Xr, Xi};

    // V = 1/4 * exp(+j theta1) exp(+j theta2) X
    V = mul_phase_plus(V, c2[k2], s2[k2]);
    V = mul_phase_plus(V, c1[k1], s1[k1]);

    V.x *= 0.25f;
    V.y *= 0.25f;

    Vh[idx2h(k1, k2, Nh)] = V;
}

// ============================================================
// DCT2D class
// ============================================================

DCT2D::DCT2D(int n1, int n2, bool use_ortho)
    : N1(n1), N2(n2), Nh(n2 / 2 + 1), ortho(use_ortho)
{
    if (N1 <= 0 || N2 <= 0)
        throw std::runtime_error("Invalid DCT2D size");

    stream = 0;

    size_t real_bytes = size_t(N1) * N2 * sizeof(float);
    size_t half_bytes = size_t(N1) * Nh * sizeof(cufftComplex);

    CHECK_CUDA(cudaMalloc(&d_v,  real_bytes));
    CHECK_CUDA(cudaMalloc(&d_Vh, half_bytes));

    CHECK_CUDA(cudaMalloc(&d_c1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c2, size_t(N2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s2, size_t(N2) * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_fwd1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fwd2, size_t(N2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inv1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inv2, size_t(N2) * sizeof(float)));

    std::vector<float> h_c1(N1);
    std::vector<float> h_s1(N1);

    for (int k = 0; k < N1; ++k)
    {
        float theta = PI_F * float(k) / (2.0f * float(N1));
        h_c1[k] = std::cos(theta);
        h_s1[k] = std::sin(theta);
    }

    CHECK_CUDA(cudaMemcpy(
        d_c1, h_c1.data(),
        size_t(N1) * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_s1, h_s1.data(),
        size_t(N1) * sizeof(float),
        cudaMemcpyHostToDevice));

    std::vector<float> h_c2(N2);
    std::vector<float> h_s2(N2);

    for (int k = 0; k < N2; ++k)
    {
        float theta = PI_F * float(k) / (2.0f * float(N2));
        h_c2[k] = std::cos(theta);
        h_s2[k] = std::sin(theta);
    }

    CHECK_CUDA(cudaMemcpy(
        d_c2, h_c2.data(),
        size_t(N2) * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_s2, h_s2.data(),
        size_t(N2) * sizeof(float),
        cudaMemcpyHostToDevice));

    std::vector<float> h_fwd1(N1), h_fwd2(N2);
    std::vector<float> h_inv1(N1), h_inv2(N2);

    for (int k = 0; k < N1; ++k)
    {
        float a = (k == 0)
            ? 1.0f / std::sqrt(float(N1))
            : std::sqrt(2.0f / float(N1));

        h_fwd1[k] = a;
        h_inv1[k] = 1.0f / a;
    }

    for (int k = 0; k < N2; ++k)
    {
        float a = (k == 0)
            ? 1.0f / std::sqrt(float(N2))
            : std::sqrt(2.0f / float(N2));

        h_fwd2[k] = a * 0.25f;
        h_inv2[k] = 4.0f / a;
    }

    CHECK_CUDA(cudaMemcpy(d_fwd1, h_fwd1.data(), size_t(N1) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fwd2, h_fwd2.data(), size_t(N2) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_inv1, h_inv1.data(), size_t(N1) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_inv2, h_inv2.data(), size_t(N2) * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUFFT(cufftPlan2d(&plan_r2c, N1, N2, CUFFT_R2C));
    CHECK_CUFFT(cufftPlan2d(&plan_c2r, N1, N2, CUFFT_C2R));

    CHECK_CUFFT(cufftSetStream(plan_r2c, stream));
    CHECK_CUFFT(cufftSetStream(plan_c2r, stream));
}

DCT2D::~DCT2D()
{
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    cudaFree(d_v);
    cudaFree(d_Vh);

    cudaFree(d_c1);
    cudaFree(d_s1);
    cudaFree(d_c2);
    cudaFree(d_s2);

    cudaFree(d_fwd1);
    cudaFree(d_fwd2);

    cudaFree(d_inv1);
    cudaFree(d_inv2);
}

void DCT2D::set_stream(cudaStream_t s)
{
    stream = s;
    cufftSetStream(plan_r2c, stream);
    cufftSetStream(plan_c2r, stream);
}

void DCT2D::forward(const float* d_in, float* d_out)
{
    dim3 block(32, 16);
    dim3 grid(div_up(N2, block.x), div_up(N1, block.y));

    k_reorder_to_fft_in_2d<<<grid, block, 0, stream>>>(
        d_in, d_v, N1, N2);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecR2C(
        plan_r2c,
        reinterpret_cast<cufftReal*>(d_v),
        reinterpret_cast<cufftComplex*>(d_Vh)));

    if (ortho)
    {
        k_dct2_forward_merge<true><<<grid, block, 0, stream>>>(
            d_Vh, d_out,
            d_c1, d_s1,
            d_c2, d_s2,
            d_fwd1, d_fwd2,
            N1, N2);
    }
    else
    {
        k_dct2_forward_merge<false><<<grid, block, 0, stream>>>(
            d_Vh, d_out,
            d_c1, d_s1,
            d_c2, d_s2,
            nullptr, nullptr,
            N1, N2);
    }

    CHECK_CUDA(cudaGetLastError());
}

void DCT2D::backward(const float* d_in, float* d_out)
{
    dim3 block(32, 16);

    dim3 grid_half(div_up(Nh, block.x), div_up(N1, block.y));
    dim3 grid_full(div_up(N2, block.x), div_up(N1, block.y));

    if (ortho)
    {
        k_idct2_build_Vh<true><<<grid_half, block, 0, stream>>>(
            d_in, d_Vh,
            d_c1, d_s1,
            d_c2, d_s2,
            d_inv1, d_inv2,
            N1, N2);
    }
    else
    {
        k_idct2_build_Vh<false><<<grid_half, block, 0, stream>>>(
            d_in, d_Vh,
            d_c1, d_s1,
            d_c2, d_s2,
            nullptr, nullptr,
            N1, N2);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecC2R(
        plan_c2r,
        reinterpret_cast<cufftComplex*>(d_Vh),
        reinterpret_cast<cufftReal*>(d_v)));

    float scale = 1.0f / float(N1 * N2);

    k_unreorder_from_fft_out_scaled_2d<<<grid_full, block, 0, stream>>>(
        d_v, d_out, N1, N2, scale);
    CHECK_CUDA(cudaGetLastError());
}