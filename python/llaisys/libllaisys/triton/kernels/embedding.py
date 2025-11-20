import torch
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None


if triton is None:
    def kernel(index, weight, out, N, D, BLOCK_SIZE=1024):
        # Fallback using torch (should not be used when Triton available)
        for i in range(N):
            idx = int(index[i].item())
            out[i].copy_(weight[idx])

else:
    @triton.jit
    def _kernel(index_ptr, weight_ptr, out_ptr, N, D, TOT, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOT

        # compute row and col for each flattened offset
        row = offsets // D
        col = offsets - row * D

        # load indices for each row
        ids = tl.load(index_ptr + row, mask=mask)

        # compute source positions in weight: ids * D + col
        src_pos = ids * D + col

        vals = tl.load(weight_ptr + src_pos, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, vals, mask=mask)


    _CACHED = None


    def kernel(index: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, N: int, D: int, BLOCK_SIZE: int = 1024):
        if triton is None:
            raise RuntimeError("Triton not available")

        global _CACHED
        if _CACHED is None:
            _CACHED = _kernel

        TOT = N * D
        grid = ((TOT + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _CACHED[grid](index, weight, out, N, D, TOT, BLOCK=BLOCK_SIZE)
