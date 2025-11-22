import triton
import triton.language as tl


@triton.jit
def _kernel(gate, up, out, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < numel

    # load as float32 for compute stability
    g = tl.load(gate + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up + offs, mask=mask, other=0.0).to(tl.float32)

    # swiglu: up * (gate / (1 + exp(-gate)))
    z = g / (1.0 + tl.exp(-g))
    out_v = u * z

    # store back (let Triton handle any necessary casting)
    tl.store(out + offs, out_v, mask=mask)


def kernel(gate, up, out, BLOCK=1024):
    # flatten inputs
    numel = gate.numel()
    grid = ( (numel + BLOCK - 1) // BLOCK, )
    _kernel[grid](gate, up, out, numel, BLOCK=BLOCK)
