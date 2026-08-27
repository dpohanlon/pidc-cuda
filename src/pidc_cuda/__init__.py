"""Fast PIDC gene-regulatory-network scoring on CPU and CUDA."""

from ._core import (
    cuda_available,
    cuda_device_name,
    pidc_cpu,
    pidc_gpu,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "cuda_available",
    "cuda_device_name",
    "pidc_cpu",
    "pidc_gpu",
]
