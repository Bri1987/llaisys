import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark


def torch_rearrange(out, inp):
    # for test we define rearrange as identity copy
    out.copy_(inp)


def test_op_rearrange(shape, dtype_name="f32", device_name="cpu", profile=False):
    print(f"   shape {shape} dtype <{dtype_name}>")
    inp, inp_ = random_tensor(shape, dtype_name, device_name)
    out, out_ = random_tensor(shape, dtype_name, device_name)

    torch_rearrange(out, inp)
    llaisys.Ops.rearrange(out_, inp_)

    assert check_equal(out_, out, strict=True)

    if profile:
        benchmark(
            lambda: torch_rearrange(out, inp),
            lambda: llaisys.Ops.rearrange(out_, inp_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    testShapes = [(2, 3), (512, 4096)]
    testDtype = ["f32", "f16", "bf16"]
    print(f"Testing Ops.rearrange on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_rearrange(shape, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
