#include "DCT3D.cuh"
#include <vector>
#include <cmath>

// ============================================================
// Reorder / unreorder
// ============================================================

__global__ void k_reorder_to_fft_in(
    const float* __restrict__ x,
    float* __restrict__ v,
    int N1, int N2, int N3)
{
    int i3 = blockIdx.x * blockDim.x + threadIdx.x;
    int i2 = blockIdx.y * blockDim.y + threadIdx.y;
    int i1 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i1 >= N1 || i2 >= N2 || i3 >= N3) return;

    int j1 = makhoul_map(i1, N1);
    int j2 = makhoul_map(i2, N2);
    int j3 = makhoul_map(i3, N3);

    v[idx3(i1, i2, i3, N2, N3)] =
        x[idx3(j1, j2, j3, N2, N3)];
}

__global__ void k_unreorder_from_fft_out_scaled(
    const float* __restrict__ v,
    float* __restrict__ x,
    int N1, int N2, int N3,
    float scale)
{
    int j3 = blockIdx.x * blockDim.x + threadIdx.x;
    int j2 = blockIdx.y * blockDim.y + threadIdx.y;
    int j1 = blockIdx.z * blockDim.z + threadIdx.z;

    if (j1 >= N1 || j2 >= N2 || j3 >= N3) return;

    int i1 = makhoul_inv_map(j1, N1);
    int i2 = makhoul_inv_map(j2, N2);
    int i3 = makhoul_inv_map(j3, N3);

    x[idx3(j1, j2, j3, N2, N3)] =
        scale * v[idx3(i1, i2, i3, N2, N3)];
}

// ============================================================
// Forward merge
// ============================================================

template <bool ORTHO>
__global__ void k_dct3_forward_merge(
    const cufftComplex* __restrict__ Vh,
    float* __restrict__ C,
    const float* __restrict__ c1,
    const float* __restrict__ s1,
    const float* __restrict__ c2,
    const float* __restrict__ s2,
    const float* __restrict__ c3,
    const float* __restrict__ s3,
    const float* __restrict__ fwd1,
    const float* __restrict__ fwd2,
    const float* __restrict__ fwd3,
    int N1, int N2, int N3)
{
    int k3 = blockIdx.x * blockDim.x + threadIdx.x;
    int k2 = blockIdx.y * blockDim.y + threadIdx.y;
    int k1 = blockIdx.z * blockDim.z + threadIdx.z;

    if (k1 >= N1 || k2 >= N2 || k3 >= N3) return;

    int Nh = N3 / 2 + 1;

    auto getV = [&](int a1, int a2, int a3) -> cufftComplex
    {
        if (a3 < Nh)
        {
            return Vh[idx3h(a1, a2, a3, N2, Nh)];
        }

        int b3 = N3 - a3;
        int b1 = mod_neg(a1, N1);
        int b2 = mod_neg(a2, N2);

        cufftComplex t = Vh[idx3h(b1, b2, b3, N2, Nh)];
        t.y = -t.y;
        return t;
    };

    int k2b = flip0(k2, N2);
    int k3b = flip0(k3, N3);

    cufftComplex V0   = getV(k1, k2,  k3);
    cufftComplex Vf3  = getV(k1, k2,  k3b);
    cufftComplex Vf2  = getV(k1, k2b, k3);
    cufftComplex Vf23 = getV(k1, k2b, k3b);

    float ca = c1[k1], sa = s1[k1];
    float cb = c2[k2], sb = s2[k2];
    float cc = c3[k3], sc = s3[k3];

    cufftComplex P0 = cadd(
        mul_phase_minus(V0,  cc, sc),
        mul_phase_plus (Vf3, cc, sc)
    );

    cufftComplex P1 = cadd(
        mul_phase_minus(Vf2,  cc, sc),
        mul_phase_plus (Vf23, cc, sc)
    );

    cufftComplex Q = cadd(
        mul_phase_minus(P0, cb, sb),
        mul_phase_plus (P1, cb, sb)
    );

    cufftComplex R = mul_phase_minus(Q, ca, sa);

    float val = 2.0f * R.x;

    if constexpr (ORTHO)
        val *= fwd1[k1] * fwd2[k2] * fwd3[k3];

    C[idx3(k1, k2, k3, N2, N3)] = val;
}

// ============================================================
// Backward merge
// ============================================================

template <bool ORTHO>
__global__ void k_idct3_build_Vh(
    const float* __restrict__ C,
    cufftComplex* __restrict__ Vh,
    const float* __restrict__ c1,
    const float* __restrict__ s1,
    const float* __restrict__ c2,
    const float* __restrict__ s2,
    const float* __restrict__ c3,
    const float* __restrict__ s3,
    const float* __restrict__ inv1,
    const float* __restrict__ inv2,
    const float* __restrict__ inv3,
    int N1, int N2, int N3)
{
    int k3 = blockIdx.x * blockDim.x + threadIdx.x;
    int k2 = blockIdx.y * blockDim.y + threadIdx.y;
    int k1 = blockIdx.z * blockDim.z + threadIdx.z;

    int Nh = N3 / 2 + 1;
    if (k1 >= N1 || k2 >= N2 || k3 >= Nh) return;

    int k3b = (k3 == 0) ? 0 : (N3 - k3);

    auto load_C = [&](int i1, int i2, int i3) -> float
    {
        float v = C[idx3(i1, i2, i3, N2, N3)];

        if constexpr (ORTHO)
            v *= inv1[i1] * inv2[i2] * inv3[i3];

        return v;
    };

    auto load_re = [&](int i1, int i2) -> float
    {
        return load_C(i1, i2, k3);
    };

    auto load_im = [&](int i1, int i2) -> float
    {
        return (k3 == 0) ? 0.0f : -load_C(i1, i2, k3b);
    };

    float Xr = load_re(k1, k2);
    float Xi = load_im(k1, k2);

    if (k1 > 0 && k2 > 0)
    {
        int k1b = N1 - k1;
        int k2b = N2 - k2;

        Xr -= load_re(k1b, k2b);
        Xi -= load_im(k1b, k2b);
    }

    if (k1 > 0)
    {
        int k1b = N1 - k1;

        Xr += load_im(k1b, k2);
        Xi -= load_re(k1b, k2);
    }

    if (k2 > 0)
    {
        int k2b = N2 - k2;

        Xr += load_im(k1, k2b);
        Xi -= load_re(k1, k2b);
    }

    cufftComplex V{Xr, Xi};

    V = mul_phase_plus(V, c3[k3], s3[k3]);
    V = mul_phase_plus(V, c2[k2], s2[k2]);
    V = mul_phase_plus(V, c1[k1], s1[k1]);

    V.x *= 0.125f;
    V.y *= 0.125f;

    Vh[idx3h(k1, k2, k3, N2, Nh)] = V;
}

// ============================================================
// DCT3D class
// ============================================================

DCT3D::DCT3D(int n1, int n2, int n3, bool use_ortho)
    : N1(n1), N2(n2), N3(n3), Nh(n3 / 2 + 1), ortho(use_ortho)
{
    if (N1 <= 0 || N2 <= 0 || N3 <= 0)
        throw std::runtime_error("Invalid DCT3D size");

    stream = 0;

    size_t real_bytes = size_t(N1) * N2 * N3 * sizeof(float);
    size_t half_bytes = size_t(N1) * N2 * Nh * sizeof(cufftComplex);

    CHECK_CUDA(cudaMalloc(&d_v,  real_bytes));
    CHECK_CUDA(cudaMalloc(&d_Vh, half_bytes));

    CHECK_CUDA(cudaMalloc(&d_c1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c2, size_t(N2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s2, size_t(N2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c3, size_t(N3) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s3, size_t(N3) * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_fwd1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fwd2, size_t(N2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fwd3, size_t(N3) * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_inv1, size_t(N1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inv2, size_t(N2) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inv3, size_t(N3) * sizeof(float)));

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

    std::vector<float> h_c3(N3);
    std::vector<float> h_s3(N3);

    for (int k = 0; k < N3; ++k)
    {
        float theta = PI_F * float(k) / (2.0f * float(N3));
        h_c3[k] = std::cos(theta);
        h_s3[k] = std::sin(theta);
    }

    CHECK_CUDA(cudaMemcpy(
        d_c3, h_c3.data(),
        size_t(N3) * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(
        d_s3, h_s3.data(),
        size_t(N3) * sizeof(float),
        cudaMemcpyHostToDevice));

    std::vector<float> h_fwd1(N1), h_fwd2(N2), h_fwd3(N3);
    std::vector<float> h_inv1(N1), h_inv2(N2), h_inv3(N3);

    for (int k = 0; k < N1; ++k)
    {
        float a = (k == 0) ? 1.0f / std::sqrt(float(N1))
                          : std::sqrt(2.0f / float(N1));
        h_fwd1[k] = a;
        h_inv1[k] = 1.0f / a;
    }

    for (int k = 0; k < N2; ++k)
    {
        float a = (k == 0) ? 1.0f / std::sqrt(float(N2))
                          : std::sqrt(2.0f / float(N2));
        h_fwd2[k] = a;
        h_inv2[k] = 1.0f / a;
    }

    for (int k = 0; k < N3; ++k)
    {
        float a = (k == 0) ? 1.0f / std::sqrt(float(N3))
                          : std::sqrt(2.0f / float(N3));

        // Forward exact old behavior:
        // val *= (a1 * a2 * a3) * 0.125f
        h_fwd3[k] = a * 0.125f;

        // Backward exact old behavior:
        // C_un = C_ortho * 8 / (a1 * a2 * a3)
        h_inv3[k] = 8.0f / a;
    }

    CHECK_CUDA(cudaMemcpy(d_fwd1, h_fwd1.data(), size_t(N1) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fwd2, h_fwd2.data(), size_t(N2) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_fwd3, h_fwd3.data(), size_t(N3) * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_inv1, h_inv1.data(), size_t(N1) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_inv2, h_inv2.data(), size_t(N2) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_inv3, h_inv3.data(), size_t(N3) * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUFFT(cufftPlan3d(&plan_r2c, N1, N2, N3, CUFFT_R2C));
    CHECK_CUFFT(cufftPlan3d(&plan_c2r, N1, N2, N3, CUFFT_C2R));

    CHECK_CUFFT(cufftSetStream(plan_r2c, stream));
    CHECK_CUFFT(cufftSetStream(plan_c2r, stream));
}

DCT3D::~DCT3D()
{
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    cudaFree(d_v);
    cudaFree(d_Vh);

    cudaFree(d_c1);
    cudaFree(d_s1);
    cudaFree(d_c2);
    cudaFree(d_s2);
    cudaFree(d_c3);
    cudaFree(d_s3);

    cudaFree(d_fwd1);
    cudaFree(d_fwd2);
    cudaFree(d_fwd3);

    cudaFree(d_inv1);
    cudaFree(d_inv2);
    cudaFree(d_inv3);
}

void DCT3D::forward(const float* d_in, float* d_out)
{
    dim3 block(32, 4, 2);
    dim3 grid(div_up(N3, block.x), div_up(N2, block.y), div_up(N1, block.z));

    k_reorder_to_fft_in<<<grid, block, 0, stream>>>(
        d_in, d_v, N1, N2, N3);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecR2C(
        plan_r2c,
        reinterpret_cast<cufftReal*>(d_v),
        reinterpret_cast<cufftComplex*>(d_Vh)));

    if (ortho)
    {
        k_dct3_forward_merge<true><<<grid, block, 0, stream>>>(
            d_Vh, d_out,
            d_c1, d_s1,
            d_c2, d_s2,
            d_c3, d_s3,
            d_fwd1, d_fwd2, d_fwd3,
            N1, N2, N3);
    }
    else
    {
        k_dct3_forward_merge<false><<<grid, block, 0, stream>>>(
            d_Vh, d_out,
            d_c1, d_s1,
            d_c2, d_s2,
            d_c3, d_s3,
            nullptr, nullptr, nullptr,
            N1, N2, N3);
    }

    CHECK_CUDA(cudaGetLastError());
}

void DCT3D::backward(const float* d_in, float* d_out)
{
    dim3 block(32, 4, 2);

    dim3 grid_half(div_up(Nh, block.x), div_up(N2, block.y), div_up(N1, block.z));
    dim3 grid_full(div_up(N3, block.x), div_up(N2, block.y), div_up(N1, block.z));

    if (ortho)
    {
        k_idct3_build_Vh<true><<<grid_half, block, 0, stream>>>(
            d_in, d_Vh,
            d_c1, d_s1,
            d_c2, d_s2,
            d_c3, d_s3,
            d_inv1, d_inv2, d_inv3,
            N1, N2, N3);
    }
    else
    {
        k_idct3_build_Vh<false><<<grid_half, block, 0, stream>>>(
            d_in, d_Vh,
            d_c1, d_s1,
            d_c2, d_s2,
            d_c3, d_s3,
            nullptr, nullptr, nullptr,
            N1, N2, N3);
    }

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecC2R(
        plan_c2r,
        reinterpret_cast<cufftComplex*>(d_Vh),
        reinterpret_cast<cufftReal*>(d_v)));

    float scale = 1.0f / float(N1 * N2 * N3);

    k_unreorder_from_fft_out_scaled<<<grid_full, block, 0, stream>>>(
        d_v, d_out, N1, N2, N3, scale);
    CHECK_CUDA(cudaGetLastError());
}