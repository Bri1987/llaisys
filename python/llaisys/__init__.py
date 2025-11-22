from .runtime import RuntimeAPI
from .libllaisys import DeviceType
from .libllaisys import DataType
from .libllaisys import MemcpyKind
from .libllaisys import llaisysStream_t as Stream
from .tensor import Tensor
from .ops import Ops
from . import models
from .models import *

__all__ = [
    "RuntimeAPI",
    "DeviceType",
    "DataType",
    "MemcpyKind",
    "Stream",
    "Tensor",
    "Ops",
    "models",
]

# Safety patch: ensure torch.masked_fill(_)/masked_fill move mask to target device
# when tests (or user code) create boolean masks on CPU but apply them to CUDA tensors.
try:
    import torch

    _orig_masked_fill = getattr(torch.Tensor, "masked_fill", None)
    _orig_masked_fill_ = getattr(torch.Tensor, "masked_fill_", None)

    if _orig_masked_fill is not None:
        def _masked_fill(self, mask, value):
            try:
                if isinstance(mask, torch.Tensor) and mask.device != self.device:
                    mask = mask.to(self.device)
            except Exception:
                pass
            return _orig_masked_fill(self, mask, value)

        torch.Tensor.masked_fill = _masked_fill

    if _orig_masked_fill_ is not None:
        def _masked_fill_(self, mask, value):
            try:
                if isinstance(mask, torch.Tensor) and mask.device != self.device:
                    mask = mask.to(self.device)
            except Exception:
                pass
            return _orig_masked_fill_(self, mask, value)

        torch.Tensor.masked_fill_ = _masked_fill_
except Exception:
    pass
