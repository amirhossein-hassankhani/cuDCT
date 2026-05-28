import torch
import torch.nn as nn

from ._cuDCT import DCT1D as _RawDCT1D
from ._cuDCT import DCT2D as _RawDCT2D
from ._cuDCT import DCT3D as _RawDCT3D


def _stream_ptr():
    return int(torch.cuda.current_stream().cuda_stream)


def _check_input(x, shape, ndim):
    if not x.is_cuda:
        raise ValueError("input must be a CUDA tensor")

    if x.dtype != torch.float32:
        raise ValueError("input must be torch.float32")

    if x.ndim != ndim:
        raise ValueError(f"input must be {ndim}D")

    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"expected shape {shape}, got {tuple(x.shape)}")

    if not x.is_contiguous():
        x = x.contiguous()

    return x


class _DCTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan, shape, ndim):
        x = _check_input(x, shape, ndim)
        y = torch.empty_like(x)

        plan.forward(x.detach(), y, _stream_ptr())

        ctx.plan = plan
        ctx.shape = tuple(shape)
        ctx.ndim = ndim

        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _check_input(grad_output, ctx.shape, ctx.ndim)
        grad_input = torch.empty_like(grad_output)

        ctx.plan.backward(grad_output.detach(), grad_input, _stream_ptr())

        return grad_input, None, None, None


class _IDCTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan, shape, ndim):
        x = _check_input(x, shape, ndim)
        y = torch.empty_like(x)

        plan.backward(x.detach(), y, _stream_ptr())

        ctx.plan = plan
        ctx.shape = tuple(shape)
        ctx.ndim = ndim

        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _check_input(grad_output, ctx.shape, ctx.ndim)
        grad_input = torch.empty_like(grad_output)

        ctx.plan.forward(grad_output.detach(), grad_input, _stream_ptr())

        return grad_input, None, None, None


class _TorchDCTBase(nn.Module):
    RawClass = None
    ndim = None
    inverse = False

    def __init__(self, shape):
        super().__init__()

        if isinstance(shape, int):
            shape = (shape,)

        self.shape = tuple(shape)

        if len(self.shape) != self.ndim:
            raise ValueError(f"shape must be {self.ndim}D, got {self.shape}")

        self.plan = self.RawClass(*self.shape, True)

    def forward(self, x):
        if self.inverse:
            return _IDCTFunction.apply(x, self.plan, self.shape, self.ndim)
        return _DCTFunction.apply(x, self.plan, self.shape, self.ndim)


class DCT1D(_TorchDCTBase):
    RawClass = _RawDCT1D
    ndim = 1
    inverse = False


class IDCT1D(_TorchDCTBase):
    RawClass = _RawDCT1D
    ndim = 1
    inverse = True


class DCT2D(_TorchDCTBase):
    RawClass = _RawDCT2D
    ndim = 2
    inverse = False


class IDCT2D(_TorchDCTBase):
    RawClass = _RawDCT2D
    ndim = 2
    inverse = True


class DCT3D(_TorchDCTBase):
    RawClass = _RawDCT3D
    ndim = 3
    inverse = False


class IDCT3D(_TorchDCTBase):
    RawClass = _RawDCT3D
    ndim = 3
    inverse = True


class DCTIDCT1D(nn.Module):
    def __init__(self, shape):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.plan = _RawDCT1D(*self.shape, True)

    def dct(self, x):
        return _DCTFunction.apply(x, self.plan, self.shape, 1)

    def idct(self, x):
        return _IDCTFunction.apply(x, self.plan, self.shape, 1)

    def forward(self, x):
        return self.dct(x)


class DCTIDCT2D(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.plan = _RawDCT2D(*self.shape, True)

    def dct(self, x):
        return _DCTFunction.apply(x, self.plan, self.shape, 2)

    def idct(self, x):
        return _IDCTFunction.apply(x, self.plan, self.shape, 2)

    def forward(self, x):
        return self.dct(x)


class DCTIDCT3D(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.plan = _RawDCT3D(*self.shape, True)

    def dct(self, x):
        return _DCTFunction.apply(x, self.plan, self.shape, 3)

    def idct(self, x):
        return _IDCTFunction.apply(x, self.plan, self.shape, 3)

    def forward(self, x):
        return self.dct(x)