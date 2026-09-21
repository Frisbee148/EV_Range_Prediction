import torch
import torch.nn as nn
import time
import os
import numpy as np
from export_to_onnx import EVRangePINN

def load_fp32_model(model_path):
    model = EVRangePINN(input_size=16)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(state_dict, nn.Module):
        model.load_state_dict(state_dict.state_dict())
    else:
        model.load_state_dict(state_dict)
    model.eval()
    # Disable gradients for inference benchmarking
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_int8_model(model_path):
    # 1. Initialize empty FP32 model
    model = EVRangePINN(input_size=16)
    # 2. Convert to quantized architecture
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )
    # 3. Load quantized weights
    quantized_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    quantized_model.eval()
    for param in quantized_model.parameters():
        param.requires_grad = False
    return quantized_model

def benchmark_model(model, name, dummy_input, iterations=500, warmup=50):
    print(f"\n--- Benchmarking {name} Model ---")
    
    # Warmup runs (to avoid counting initialization overhead)
    for _ in range(warmup):
        _ = model(dummy_input)
        
    # Benchmark runs
    latencies = []
    
    start_time_total = time.time()
    for _ in range(iterations):
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0) # convert to ms
        
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = iterations / total_time
    
    print(f"Iterations: {iterations}")
    print(f"Mean Latency:  {mean_latency:.2f} ms")
    print(f"99th Percentile: {p99_latency:.2f} ms")
    print(f"Throughput:    {throughput:.2f} predictions/sec")
    
    return mean_latency, throughput

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fp32_path = os.path.join(current_dir, "15_07_model.pth")
    int8_path = os.path.join(current_dir, "ev_pinn_int8.pth")
    
    # Enable multithreading optimizations common on edge CPUs
    torch.set_num_threads(4) 
    
    print("Loading models...")
    fp32_model = load_fp32_model(fp32_path)
    int8_model = load_int8_model(int8_path)
    
    # Create dummy input tensor simulating 1 vehicle streaming 10 historical timesteps
    # Shape: (batch_size=1, sequence_length=10, features=16)
    dummy_input = torch.randn(1, 10, 16)
    
    print("\nStarting Edge Inference Simulation...")
    print("Simulating on CPU (typical edge vehicle computer constraint)")
    
    fp32_lat, fp32_thr = benchmark_model(fp32_model, "Original (FP32)", dummy_input)
    int8_lat, int8_thr = benchmark_model(int8_model, "Quantized (INT8)", dummy_input)
    
    print("\n================ FINAL REPORT ================")
    print(f"Latency Speedup: {fp32_lat / int8_lat:.2f}x faster per prediction")
    print(f"Throughput Boost: {int8_thr / fp32_thr:.2f}x more predictions per second")
    print("==============================================")
