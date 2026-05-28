from ._cuDCT import DCT1D as _RawDCT1D
from ._cuDCT import DCT2D as _RawDCT2D
from ._cuDCT import DCT3D as _RawDCT3D


def _stream_ptr(stream=None):
    if stream is None:
        return 0

    if isinstance(stream, int):
        return stream

    if hasattr(stream, "ptr"):
        return int(stream.ptr)

    if hasattr(stream, "cuda_stream"):
        return int(stream.cuda_stream)

    raise TypeError("stream must be None, int, or an object with .ptr/.cuda_stream")


class _NativeBase:
    RawClass = None
    ndim = None

    def __init__(self, shape, ortho=True):
        if isinstance(shape, int):
            shape = (shape,)

        self.shape = tuple(shape)
        self.ortho = bool(ortho)

        if len(self.shape) != self.ndim:
            raise ValueError(f"shape must be {self.ndim}D, got {self.shape}")

        self.obj = self.RawClass(*self.shape, self.ortho)

    def forward(self, x, out, stream=None):
        """
        x and out can be any CUDA tensor/array supporting __dlpack__:
        CuPy, PyTorch, JAX, etc.

        out is required because native.py does not depend on a tensor library.
        """
        self.obj.forward(x, out, _stream_ptr(stream))
        return out

    def backward(self, x, out, stream=None):
        """
        x and out can be any CUDA tensor/array supporting __dlpack__.
        """
        self.obj.backward(x, out, _stream_ptr(stream))
        return out


class DCT1D(_NativeBase):
    RawClass = _RawDCT1D
    ndim = 1


class DCT2D(_NativeBase):
    RawClass = _RawDCT2D
    ndim = 2


class DCT3D(_NativeBase):
    RawClass = _RawDCT3D
    ndim = 3