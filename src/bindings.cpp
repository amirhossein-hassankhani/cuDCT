#include <pybind11/pybind11.h>

#include <Python.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "dlpack.h"

#include "DCT1D.cuh"
#include "DCT2D.cuh"
#include "DCT3D.cuh"

namespace py = pybind11;

// ============================================================
// DLPack tensor view
// ============================================================

struct DlTensorView {
    float* data = nullptr;
    int ndim = 0;
    int64_t n1 = 1;
    int64_t n2 = 1;
    int64_t n3 = 1;
};

// ============================================================
// DLPack helpers
// ============================================================

static py::capsule get_dlpack_capsule(py::object obj, uintptr_t stream)
{
    if (!py::hasattr(obj, "__dlpack__"))
        throw std::runtime_error("Object does not support __dlpack__");

    if (stream == 0)
        return obj.attr("__dlpack__")().cast<py::capsule>();

    return obj.attr("__dlpack__")(py::arg("stream") = stream).cast<py::capsule>();
}

static DLManagedTensor* get_managed_tensor(py::capsule cap)
{
    const char* name = PyCapsule_GetName(cap.ptr());

    if (!name || std::string(name) != "dltensor")
        throw std::runtime_error("Invalid DLPack capsule name");

    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(cap.ptr(), "dltensor")
    );

    if (!managed)
        throw std::runtime_error("Could not read DLPack capsule");

    return managed;
}

static void validate_common_dlpack(const DLTensor& t)
{
    if (t.device.device_type != kDLCUDA &&
        t.device.device_type != kDLCUDAManaged)
    {
        throw std::runtime_error("Tensor must be on CUDA device");
    }

    if (t.dtype.code != kDLFloat || t.dtype.bits != 32 || t.dtype.lanes != 1)
        throw std::runtime_error("Tensor must be float32");
}

static float* dlpack_data_ptr(const DLTensor& t)
{
    char* base = static_cast<char*>(t.data);
    return reinterpret_cast<float*>(base + t.byte_offset);
}

static DlTensorView parse_cuda_float32(py::capsule cap, int expected_ndim)
{
    DLManagedTensor* managed = get_managed_tensor(cap);
    DLTensor& t = managed->dl_tensor;

    validate_common_dlpack(t);

    if (t.ndim != expected_ndim)
    {
        throw std::runtime_error(
            "Tensor ndim mismatch: expected " +
            std::to_string(expected_ndim) +
            ", got " +
            std::to_string(t.ndim)
        );
    }

    DlTensorView v;
    v.data = dlpack_data_ptr(t);
    v.ndim = t.ndim;

    if (expected_ndim == 1)
    {
        v.n1 = t.shape[0];

        if (t.strides && t.strides[0] != 1)
            throw std::runtime_error("1D tensor must be contiguous");
    }
    else if (expected_ndim == 2)
    {
        v.n1 = t.shape[0];
        v.n2 = t.shape[1];

        if (t.strides)
        {
            if (t.strides[1] != 1 ||
                t.strides[0] != v.n2)
            {
                throw std::runtime_error("2D tensor must be C-contiguous");
            }
        }
    }
    else if (expected_ndim == 3)
    {
        v.n1 = t.shape[0];
        v.n2 = t.shape[1];
        v.n3 = t.shape[2];

        if (t.strides)
        {
            if (t.strides[2] != 1 ||
                t.strides[1] != v.n3 ||
                t.strides[0] != v.n2 * v.n3)
            {
                throw std::runtime_error("3D tensor must be C-contiguous");
            }
        }
    }
    else
    {
        throw std::runtime_error("Only 1D, 2D, and 3D tensors are supported");
    }

    return v;
}

// ============================================================
// PyDCT1D
// ============================================================

class PyDCT1D {
public:
    PyDCT1D(int n, bool ortho)
        : n_(n), impl_(n, ortho)
    {}

    void set_stream(uintptr_t stream_ptr)
    {
        impl_.set_stream(reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    void forward(py::object x, py::object y, uintptr_t stream_ptr = 0)
    {
        auto x_cap = get_dlpack_capsule(x, stream_ptr);
        auto y_cap = get_dlpack_capsule(y, stream_ptr);

        auto xv = parse_cuda_float32(x_cap, 1);
        auto yv = parse_cuda_float32(y_cap, 1);

        check_shape(xv);
        check_shape(yv);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
        impl_.set_stream(s);

        py::gil_scoped_release release;
        impl_.forward(xv.data, yv.data);
    }

    void backward(py::object x, py::object y, uintptr_t stream_ptr = 0)
    {
        auto x_cap = get_dlpack_capsule(x, stream_ptr);
        auto y_cap = get_dlpack_capsule(y, stream_ptr);

        auto xv = parse_cuda_float32(x_cap, 1);
        auto yv = parse_cuda_float32(y_cap, 1);

        check_shape(xv);
        check_shape(yv);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
        impl_.set_stream(s);

        py::gil_scoped_release release;
        impl_.backward(xv.data, yv.data);
    }

private:
    void check_shape(const DlTensorView& v) const
    {
        if (v.n1 != n_)
            throw std::runtime_error("Tensor shape does not match DCT1D object shape");
    }

    int n_;
    DCT1D impl_;
};

// ============================================================
// PyDCT2D
// ============================================================

class PyDCT2D {
public:
    PyDCT2D(int n1, int n2, bool ortho)
        : n1_(n1), n2_(n2), impl_(n1, n2, ortho)
    {}

    void set_stream(uintptr_t stream_ptr)
    {
        impl_.set_stream(reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    void forward(py::object x, py::object y, uintptr_t stream_ptr = 0)
    {
        auto x_cap = get_dlpack_capsule(x, stream_ptr);
        auto y_cap = get_dlpack_capsule(y, stream_ptr);

        auto xv = parse_cuda_float32(x_cap, 2);
        auto yv = parse_cuda_float32(y_cap, 2);

        check_shape(xv);
        check_shape(yv);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
        impl_.set_stream(s);

        py::gil_scoped_release release;
        impl_.forward(xv.data, yv.data);
    }

    void backward(py::object x, py::object y, uintptr_t stream_ptr = 0)
    {
        auto x_cap = get_dlpack_capsule(x, stream_ptr);
        auto y_cap = get_dlpack_capsule(y, stream_ptr);

        auto xv = parse_cuda_float32(x_cap, 2);
        auto yv = parse_cuda_float32(y_cap, 2);

        check_shape(xv);
        check_shape(yv);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
        impl_.set_stream(s);

        py::gil_scoped_release release;
        impl_.backward(xv.data, yv.data);
    }

private:
    void check_shape(const DlTensorView& v) const
    {
        if (v.n1 != n1_ || v.n2 != n2_)
            throw std::runtime_error("Tensor shape does not match DCT2D object shape");
    }

    int n1_;
    int n2_;
    DCT2D impl_;
};

// ============================================================
// PyDCT3D
// ============================================================

class PyDCT3D {
public:
    PyDCT3D(int n1, int n2, int n3, bool ortho)
        : n1_(n1), n2_(n2), n3_(n3), impl_(n1, n2, n3, ortho)
    {}

    void set_stream(uintptr_t stream_ptr)
    {
        impl_.set_stream(reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    void forward(py::object x, py::object y, uintptr_t stream_ptr = 0)
    {
        auto x_cap = get_dlpack_capsule(x, stream_ptr);
        auto y_cap = get_dlpack_capsule(y, stream_ptr);

        auto xv = parse_cuda_float32(x_cap, 3);
        auto yv = parse_cuda_float32(y_cap, 3);

        check_shape(xv);
        check_shape(yv);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
        impl_.set_stream(s);

        py::gil_scoped_release release;
        impl_.forward(xv.data, yv.data);
    }

    void backward(py::object x, py::object y, uintptr_t stream_ptr = 0)
    {
        auto x_cap = get_dlpack_capsule(x, stream_ptr);
        auto y_cap = get_dlpack_capsule(y, stream_ptr);

        auto xv = parse_cuda_float32(x_cap, 3);
        auto yv = parse_cuda_float32(y_cap, 3);

        check_shape(xv);
        check_shape(yv);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
        impl_.set_stream(s);

        py::gil_scoped_release release;
        impl_.backward(xv.data, yv.data);
    }

private:
    void check_shape(const DlTensorView& v) const
    {
        if (v.n1 != n1_ || v.n2 != n2_ || v.n3 != n3_)
            throw std::runtime_error("Tensor shape does not match DCT3D object shape");
    }

    int n1_;
    int n2_;
    int n3_;
    DCT3D impl_;
};

// ============================================================
// Module
// ============================================================

PYBIND11_MODULE(_cuDCT, m)
{
    py::class_<PyDCT1D>(m, "DCT1D")
        .def(py::init<int, bool>(),
             py::arg("n"),
             py::arg("ortho") = true)
        .def("set_stream", &PyDCT1D::set_stream)
        .def("forward", &PyDCT1D::forward,
             py::arg("x"),
             py::arg("out"),
             py::arg("stream") = 0)
        .def("backward", &PyDCT1D::backward,
             py::arg("x"),
             py::arg("out"),
             py::arg("stream") = 0);

    py::class_<PyDCT2D>(m, "DCT2D")
        .def(py::init<int, int, bool>(),
             py::arg("n1"),
             py::arg("n2"),
             py::arg("ortho") = true)
        .def("set_stream", &PyDCT2D::set_stream)
        .def("forward", &PyDCT2D::forward,
             py::arg("x"),
             py::arg("out"),
             py::arg("stream") = 0)
        .def("backward", &PyDCT2D::backward,
             py::arg("x"),
             py::arg("out"),
             py::arg("stream") = 0);

    py::class_<PyDCT3D>(m, "DCT3D")
        .def(py::init<int, int, int, bool>(),
             py::arg("n1"),
             py::arg("n2"),
             py::arg("n3"),
             py::arg("ortho") = true)
        .def("set_stream", &PyDCT3D::set_stream)
        .def("forward", &PyDCT3D::forward,
             py::arg("x"),
             py::arg("out"),
             py::arg("stream") = 0)
        .def("backward", &PyDCT3D::backward,
             py::arg("x"),
             py::arg("out"),
             py::arg("stream") = 0);
}