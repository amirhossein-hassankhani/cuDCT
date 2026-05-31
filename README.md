# cuDCT

[![Build cuDCT](https://github.com/amirhossein-hassankhani/cuDCT/actions/workflows/build.yml/badge.svg)](https://github.com/amirhossein-hassankhani/cuDCT/actions/workflows/build.yml)

Fast CUDA/cuFFT-based DCT-II and DCT-III (IDCT) for 1D, 2D, and 3D tensors.

`cuDCT` implements orthonormal discrete cosine transforms directly on the GPU using FFT-based algorithms derived from the Makhoul method. The library is designed for tensor workflows and exposes zero-copy DLPack bindings. There is already a PyTorch wrapper class with autograd support. This implementation is ideal for cases where the size is fixed but the conetent change during each iteration. 

---

## Features

- GPU DCT-II / IDCT
- 1D, 2D, and 3D transforms
- PyTorch autograd support
- CuPy / DLPack native interface
- cuFFT-based implementation
- Lower memory usage than separable DCT implementations
- Orthogonal normalization, equivalent to `norm="ortho"`

---

## Algorithm

cuDCT uses an FFT-based DCT construction based on the method introduced by Makhoul.

Instead of explicitly building an even extension of length `2N`, the input is reordered into a compact FFT-friendly sequence:

```text
x = [x0, x1, x2, x3, x4, x5, ...]

v = [x0, x2, x4, ..., x5, x3, x1]
```

A real-to-complex FFT is then applied, followed by a phase correction:

```math
e^{-j \pi k / (2N)}
```

to recover the DCT coefficients.

For multidimensional transforms, cuDCT applies the Makhoul mapping along each axis, runs one multidimensional cuFFT, and uses fused merge kernels. This avoids the repeated 1D transforms, tensor transposes, and intermediate allocations used by separable implementations.

Reference:

- J. Makhoul, *A Fast Cosine Transform in One and Two Dimensions*, IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.

---

## Installation

```bash
pip install .
```

---

## PyTorch Example

```python
import torch
from cuDCT.torch import DCT3D, IDCT3D

x = torch.randn(64, 64, 64, device="cuda")

dct = DCT3D(x.shape).cuda()
idct = IDCT3D(x.shape).cuda()

y = dct(x)
z = idct(y)

print((x - z).abs().max())
```

---

## CuPy Example

```python
import cupy as cp
from cuDCT.native import DCT3D

x = cp.random.randn(64, 64, 64).astype(cp.float32)
y = cp.empty_like(x)

dct = DCT3D(x.shape)

dct.forward(
    x,
    y,
    stream=cp.cuda.get_current_stream()
)
```

---

## Benchmark

Benchmarks use FP32 tensors with orthonormal normalization.

References:

- GPU reference: `torch-dct`
- CPU reference: `scipy.fft`

Typical relative error:

```text
~1e-7 to ~1e-6
```

---

# NVIDIA A100 Results

## 1D Performance

| Size | cuDCT Fwd | cuDCT Inv | torch-dct Fwd | torch-dct Inv | vs torch-dct | SciPy CPU Fwd | SciPy CPU Inv | vs SciPy CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65K | 0.075 ms | 0.071 ms | 0.387 ms | 0.553 ms | 5–8× | 0.610 ms | 0.569 ms | 8× |
| 1M | 0.073 ms | 0.071 ms | 0.370 ms | 0.571 ms | 5–8× | 14.530 ms | 13.795 ms | ~200× |
| 16M | 0.839 ms | 0.701 ms | 2.212 ms | 3.580 ms | 2–5× | 367.770 ms | 360.674 ms | 440–515× |
| 67M | 3.534 ms | 2.984 ms | 8.767 ms | 14.264 ms | 2–5× | skipped | skipped | - |

## 2D Performance

| Size | cuDCT Fwd | cuDCT Inv | torch-dct Fwd | torch-dct Inv | vs torch-dct | SciPy CPU Fwd | SciPy CPU Inv | vs SciPy CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64² | 0.065 ms | 0.065 ms | 0.742 ms | 1.184 ms | 11–18× | 0.047 ms | 0.039 ms | <1× |
| 256² | 0.065 ms | 0.067 ms | 0.748 ms | 1.157 ms | 11–17× | 0.359 ms | 0.352 ms | ~5× |
| 1024² | 0.068 ms | 0.067 ms | 0.766 ms | 1.221 ms | 11–18× | 7.776 ms | 7.329 ms | ~110× |
| 4096² | 0.569 ms | 0.562 ms | 2.562 ms | 5.540 ms | 4–10× | 257.768 ms | 224.507 ms | 400–450× |

## 3D Performance

| Size | cuDCT Fwd | cuDCT Inv | torch-dct Fwd | torch-dct Inv | vs torch-dct | SciPy CPU Fwd | SciPy CPU Inv | vs SciPy CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32³ | 0.069 ms | 0.069 ms | 1.115 ms | 1.684 ms | 16–24× | 0.326 ms | 0.316 ms | 4–5× |
| 128³ | 0.101 ms | 0.096 ms | 1.120 ms | 1.729 ms | 11–18× | 19.648 ms | 19.345 ms | ~200× |
| 256³ | 0.638 ms | 0.638 ms | 4.341 ms | 8.790 ms | 7–14× | 200.321 ms | 211.234 ms | 310–330× |
| 512³ | 5.709 ms | 5.054 ms | 35.191 ms | 75.113 ms | 6–15× | skipped | skipped | - |

---

# NVIDIA T4 Results

The T4 is significantly weaker than the A100, but cuDCT still outperforms `torch-dct` and SciPy for large transforms.

## 1D Performance

| Size | cuDCT Fwd | cuDCT Inv | torch-dct Fwd | torch-dct Inv | vs torch-dct | SciPy CPU Fwd | SciPy CPU Inv | vs SciPy CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65K | 0.071 ms | 0.070 ms | 0.357 ms | 0.562 ms | 5–8× | 0.551 ms | 0.547 ms | ~8× |
| 1M | 0.339 ms | 0.300 ms | 0.841 ms | 1.296 ms | 2–4× | 13.317 ms | 12.444 ms | 39–42× |
| 16M | 4.955 ms | 4.240 ms | 11.659 ms | 19.501 ms | 2–5× | 494.764 ms | 355.617 ms | 84–100× |
| 67M | 20.444 ms | 17.367 ms | 47.113 ms | 78.542 ms | 2–5× | skipped | skipped | - |

## 2D Performance

| Size | cuDCT Fwd | cuDCT Inv | torch-dct Fwd | torch-dct Inv | vs torch-dct | SciPy CPU Fwd | SciPy CPU Inv | vs SciPy CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64² | 0.062 ms | 0.060 ms | 0.736 ms | 1.163 ms | 12–19× | 0.037 ms | 0.050 ms | <1× |
| 256² | 0.061 ms | 0.061 ms | 0.928 ms | 1.099 ms | 15–18× | 0.330 ms | 0.319 ms | ~5× |
| 1024² | 0.159 ms | 0.162 ms | 0.853 ms | 1.897 ms | 5–12× | 9.184 ms | 6.427 ms | 40–58× |
| 4096² | 3.337 ms | 3.202 ms | 12.900 ms | 29.392 ms | 4–9× | 262.437 ms | 208.389 ms | 65–79× |

## 3D Performance

| Size | cuDCT Fwd | cuDCT Inv | torch-dct Fwd | torch-dct Inv | vs torch-dct | SciPy CPU Fwd | SciPy CPU Inv | vs SciPy CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32³ | 0.065 ms | 0.064 ms | 1.313 ms | 1.698 ms | 20–27× | 0.274 ms | 0.271 ms | ~4× |
| 128³ | 0.530 ms | 0.543 ms | 2.707 ms | 5.778 ms | 5–11× | 17.394 ms | 17.429 ms | ~32× |
| 256³ | 3.699 ms | 3.619 ms | 27.933 ms | 59.058 ms | 8–16× | 201.975 ms | 201.565 ms | ~55× |
| 512³ | 30.227 ms | 29.577 ms | 224.210 ms | 474.776 ms | 7–16× | skipped | skipped | - |

---

## Memory Usage

cuDCT generally uses substantially less memory than `torch-dct`.

Example: `512³`

| cuDCT Forward Memory | torch-dct Forward Memory |
|---|---:|
| 0.500 GiB | 4.754 GiB |

Example: `4096²`

| cuDCT Forward Memory | torch-dct Forward Memory |
|---|---:|
| 0.062 GiB | 0.535 GiB |



---

cuDCT is fastest when the transform is large enough that GPU launch overhead is amortized.

For small CPU-sized 2D cases such as `64²`, SciPy can be faster because GPU launch overhead dominates. For larger 1D, 2D, and 3D transforms, cuDCT is substantially faster than SciPy CPU.

Compared with `torch-dct`, cuDCT is faster because `torch-dct` performs multidimensional transforms using separable 1D passes. That requires repeated FFTs, transposes, and intermediate tensors. cuDCT instead uses one multidimensional cuFFT plus fused mapping and merge kernels.

---

## Project Structure

```text
src/
    DCT1D.cu
    DCT2D.cu
    DCT3D.cu
    DCTCommon.cuh
    bindings.cpp

python/
    cuDCT/
        native.py
        torch.py
```

---

## TODO

- FP64 / double precision support
- Batched transforms
- Multi-GPU support


