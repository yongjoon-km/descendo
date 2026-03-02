import time
from typing import Sequence, TypedDict

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


class Latency(TypedDict):
    avg_latency_ms: float
    max_latency_ms: float


class Benchmark(TypedDict):
    runs: int
    mean_absolute_difference: float
    pytorch: Latency
    executorch: Latency


def benchmark_model(
    pytorch_model: torch.nn.Module,
    input_shape: Sequence[int],
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Benchmark:
    pytorch_model.eval()

    sample_input = (torch.randn(input_shape),)

    exported_program = torch.export.export(pytorch_model, sample_input)

    et_program = to_edge_transform_and_lower(
                exported_program,
                partitioner = []
    ).to_executorch()

    runtime = Runtime.get()
    program = runtime.load_program(et_program.buffer)
    method = program.load_method("forward")
    if method is None:
        raise Exception("method should exist")

    for _ in range(warmup_runs):
        dummy_input = (torch.randn(input_shape),)
        with torch.no_grad():
            pytorch_model(*dummy_input)
        _ = method.execute(list(dummy_input))

    pt_latencies: list[float] = []
    et_latencies: list[float] = []
    mad_values: list[float] = []

    for _ in range(num_runs):
        test_input = (torch.randn(input_shape),)

        t0 = time.perf_counter()
        with torch.no_grad():
            pt_out = pytorch_model(*test_input)
        t1 = time.perf_counter()
        pt_latencies.append((t1 - t0) * 1000)

        t2 = time.perf_counter()
        et_out = method.execute(list(test_input))[0]
        t3 = time.perf_counter()
        et_latencies.append((t3 - t2) * 1000)

        diff_tensor = torch.abs(pt_out - et_out)
        mad_values.append(diff_tensor.mean().item())

    return {
        "runs": num_runs,
        "mean_absolute_difference": sum(mad_values) / num_runs,
        "pytorch": {
            "avg_latency_ms": sum(pt_latencies) / num_runs,
            "max_latency_ms": max(pt_latencies)
        },
        "executorch": {
            "avg_latency_ms": sum(et_latencies) / num_runs,
            "max_latency_ms": max(et_latencies)
        }
    }
