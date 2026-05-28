import time
import math
import gc

import torch
import torch_dct as tdct
from scipy.fft import dct, idct, dctn, idctn

from cuDCT.torch import (
    DCTIDCT1D,
    DCTIDCT2D,
    DCTIDCT3D,
)


def cuda_time(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def cpu_time(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    t1 = time.perf_counter()

    return 1000.0 * (t1 - t0) / repeat


def gpu_free_gib():
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def torch_mem_gib():
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved": torch.cuda.max_memory_reserved() / 1024**3,
    }


def measure_gpu_memory(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    free_before, total = gpu_free_gib()
    torch_before = torch_mem_gib()

    out = fn()

    torch.cuda.synchronize()

    free_after, _ = gpu_free_gib()
    torch_after = torch_mem_gib()

    return {
        "out": out,
        "device_used_delta_gib": max(0.0, free_before - free_after),
        "torch_alloc_delta_gib": max(0.0, torch_after["allocated"] - torch_before["allocated"]),
        "torch_reserved_delta_gib": max(0.0, torch_after["reserved"] - torch_before["reserved"]),
        "torch_peak_alloc_gib": torch_after["max_allocated"],
        "torch_peak_reserved_gib": torch_after["max_reserved"],
        "device_total_gib": total,
    }


def torch_dct_forward(x, ndim):
    if ndim == 1:
        return tdct.dct(x, norm="ortho")
    if ndim == 2:
        return tdct.dct_2d(x, norm="ortho")
    if ndim == 3:
        return tdct.dct_3d(x, norm="ortho")
    raise ValueError("ndim must be 1, 2, or 3")


def torch_dct_inverse(x, ndim):
    if ndim == 1:
        return tdct.idct(x, norm="ortho")
    if ndim == 2:
        return tdct.idct_2d(x, norm="ortho")
    if ndim == 3:
        return tdct.idct_3d(x, norm="ortho")
    raise ValueError("ndim must be 1, 2, or 3")


def scipy_forward(x_cpu, ndim):
    if ndim == 1:
        return dct(x_cpu, type=2, norm="ortho")
    return dctn(x_cpu, type=2, norm="ortho")


def scipy_inverse(y_cpu, ndim):
    if ndim == 1:
        return idct(y_cpu, type=2, norm="ortho")
    return idctn(y_cpu, type=2, norm="ortho")


def rel_error(a, b):
    abs_err = (a - b).abs().max()
    denom = b.abs().max().clamp_min(1e-30)
    return abs_err.item(), (abs_err / denom).item()


def tensor_gib(shape):
    return math.prod(shape) * 4.0 / 1024**3


def repeat_for_size(numel):
    if numel <= 2**18:
        return 100
    if numel <= 2**22:
        return 50
    if numel <= 2**25:
        return 20
    return 5


def should_do_cpu(numel):
    return numel <= 16_777_216


def should_do_torch_dct(numel):
    return numel <= 134_217_728


def print_mem(label, mem):
    print(
        f"{label} memory: device_delta={mem['device_used_delta_gib']:.3f} GiB "
        f"| torch_peak_alloc={mem['torch_peak_alloc_gib']:.3f} GiB "
        f"| torch_peak_reserved={mem['torch_peak_reserved_gib']:.3f} GiB"
    )


def run_one(name, ndim, shape, Module, seed):
    numel = math.prod(shape)
    repeat = repeat_for_size(numel)

    print(f"\n{name} shape={shape} | elems={numel:,} | tensor={tensor_gib(shape):.3f} GiB")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)

        # cuDCT plan memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_before_plan, _ = gpu_free_gib()

        cuDCT = Module(shape)

        torch.cuda.synchronize()
        free_after_plan, _ = gpu_free_gib()

        plan_mem = max(0.0, free_before_plan - free_after_plan)

        y = cuDCT.dct(x)
        z = cuDCT.idct(y)
        torch.cuda.synchronize()

        round_abs, round_rel = rel_error(z, x)

        t_fwd = cuda_time(lambda: cuDCT.dct(x), repeat=repeat)
        t_inv = cuda_time(lambda: cuDCT.idct(y), repeat=repeat)
        t_round = cuda_time(lambda: cuDCT.idct(cuDCT.dct(x)), repeat=repeat)

        mem_cudct_fwd = measure_gpu_memory(lambda: cuDCT.dct(x))
        mem_cudct_inv = measure_gpu_memory(lambda: cuDCT.idct(y))
        mem_cudct_round = measure_gpu_memory(lambda: cuDCT.idct(cuDCT.dct(x)))

        print(f"cuDCT time: fwd={t_fwd:.3f} ms | inv={t_inv:.3f} ms | round={t_round:.3f} ms")
        print(f"cuDCT plan memory: {plan_mem:.3f} GiB")
        print_mem("cuDCT forward", mem_cudct_fwd)
        print_mem("cuDCT inverse", mem_cudct_inv)
        print_mem("cuDCT roundtrip", mem_cudct_round)
        print(f"cuDCT roundtrip error: abs={round_abs:.3e} | rel={round_rel:.3e}")

        if should_do_torch_dct(numel):
            y_ref = torch_dct_forward(x, ndim)
            z_ref = torch_dct_inverse(y_ref, ndim)
            torch.cuda.synchronize()

            f_abs, f_rel = rel_error(y, y_ref)
            i_abs, i_rel = rel_error(z, z_ref)

            t_ref_fwd = cuda_time(lambda: torch_dct_forward(x, ndim), repeat=repeat)
            t_ref_inv = cuda_time(lambda: torch_dct_inverse(y_ref, ndim), repeat=repeat)

            mem_ref_fwd = measure_gpu_memory(lambda: torch_dct_forward(x, ndim))
            mem_ref_inv = measure_gpu_memory(lambda: torch_dct_inverse(y_ref, ndim))

            print(
                f"torch-dct time: fwd={t_ref_fwd:.3f} ms | inv={t_ref_inv:.3f} ms "
                f"| speedup fwd={t_ref_fwd / t_fwd:.2f}x | inv={t_ref_inv / t_inv:.2f}x"
            )
            print_mem("torch-dct forward", mem_ref_fwd)
            print_mem("torch-dct inverse", mem_ref_inv)
            print(
                f"vs torch-dct accuracy: fwd abs={f_abs:.3e}, rel={f_rel:.3e} "
                f"| inv abs={i_abs:.3e}, rel={i_rel:.3e}"
            )
        else:
            print("torch-dct: skipped for size")

        if should_do_cpu(numel):
            x_cpu = x.detach().cpu().numpy()

            y_cpu = scipy_forward(x_cpu, ndim)
            z_cpu = scipy_inverse(y_cpu, ndim)

            y_cpu_t = torch.from_numpy(y_cpu).to(device="cuda", dtype=torch.float32)
            z_cpu_t = torch.from_numpy(z_cpu).to(device="cuda", dtype=torch.float32)

            f_abs, f_rel = rel_error(y, y_cpu_t)
            i_abs, i_rel = rel_error(z, z_cpu_t)

            t_cpu_fwd = cpu_time(lambda: scipy_forward(x_cpu, ndim))
            t_cpu_inv = cpu_time(lambda: scipy_inverse(y_cpu, ndim))

            print(
                f"SciPy CPU time: fwd={t_cpu_fwd:.3f} ms | inv={t_cpu_inv:.3f} ms "
                f"| speedup fwd={t_cpu_fwd / t_fwd:.2f}x | inv={t_cpu_inv / t_inv:.2f}x"
            )
            print(
                f"vs SciPy accuracy: fwd abs={f_abs:.3e}, rel={f_rel:.3e} "
                f"| inv abs={i_abs:.3e}, rel={i_rel:.3e}"
            )
        else:
            print("SciPy CPU: skipped for size")

    except torch.cuda.OutOfMemoryError:
        print("OOM: skipped")
        torch.cuda.empty_cache()

    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tests = [
        ("1D", 1, DCTIDCT1D, [
            (2**16,),
            (2**18,),
            (2**20,),
            (2**22,),
            (2**24,),
            (2**26,),
        ]),
        ("2D", 2, DCTIDCT2D, [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ]),
        ("3D", 3, DCTIDCT3D, [
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]),
    ]

    for seed, (name, ndim, module, shapes) in enumerate(tests):
        print("\n" + "=" * 100)
        print(f"{name} BENCHMARK")
        print("=" * 100)

        for shape in shapes:
            run_one(name, ndim, shape, module, seed)


if __name__ == "__main__":
    main()