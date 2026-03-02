from task.benchmark import benchmark_model
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x))


def test_benchmark():
    model = SimpleModel()
    
    benchmark = benchmark_model(model, input_shape=(1, 10), num_runs=5, warmup_runs=2)
    
    print("\n--- Benchmark Results ---")
    print(f"Overall MAD: {benchmark['mean_absolute_difference']:.8f}")
    print(f"PyTorch    - Avg: {benchmark['pytorch']['avg_latency_ms']:.4f} ms | Max: {benchmark['pytorch']['max_latency_ms']:.4f} ms")
    print(f"ExecuTorch - Avg: {benchmark['executorch']['avg_latency_ms']:.4f} ms | Max: {benchmark['executorch']['max_latency_ms']:.4f} ms")

    assert "runs" in benchmark, "Missing 'runs' field"
    assert "mean_absolute_difference" in benchmark, "Missing 'mean_absolute_difference' field"
    assert "pytorch" in benchmark, "Missing 'pytorch' field"
    assert "executorch" in benchmark, "Missing 'executorch' field"

    assert "avg_latency_ms" in benchmark["pytorch"], "Missing PyTorch avg_latency"
    assert "max_latency_ms" in benchmark["pytorch"], "Missing PyTorch max_latency"
    assert "avg_latency_ms" in benchmark["executorch"], "Missing ExecuTorch avg_latency"
    assert "max_latency_ms" in benchmark["executorch"], "Missing ExecuTorch max_latency"

    assert benchmark["runs"] == 5
    assert benchmark["mean_absolute_difference"] >= 0.0  # MAD cannot be negative
    
    assert benchmark["pytorch"]["avg_latency_ms"] > 0.0
    assert benchmark["executorch"]["avg_latency_ms"] > 0.0
