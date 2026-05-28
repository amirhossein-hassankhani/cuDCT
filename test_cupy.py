import numpy as np
import cupy as cp
from scipy.fft import dctn

from cuDCT.native import DCT3D

x = cp.random.randn(64, 64, 64).astype(cp.float32)
y = cp.empty_like(x)

dct = DCT3D(x.shape, ortho=True)
dct.forward(x, y, stream=cp.cuda.get_current_stream())

cp.cuda.Stream.null.synchronize()

x_np = cp.asnumpy(x)
y_np = cp.asnumpy(y)

y_ref = dctn(x_np, type=2, norm="ortho").astype(np.float32)

abs_err = np.max(np.abs(y_np - y_ref))
rel_err = abs_err / np.max(np.abs(y_ref))

print("max abs error:", abs_err)
print("relative error:", rel_err)
print("allclose:", np.allclose(y_np, y_ref, rtol=1e-4, atol=1e-4))