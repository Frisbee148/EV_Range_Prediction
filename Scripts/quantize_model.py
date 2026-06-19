import torch
import torch.nn as nn
import os
from export_to_onnx import EVRangePINN

def quantize_pytorch_model(model_path, output_path):
    print(f"Loading PyTorch model for quantization from: {model_path}")
    
    # 1. Initialize the FP32 model
    model = EVRangePINN(input_size=16)
    
    # 2. Load the FP32 weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(state_dict, nn.Module):
        model.load_state_dict(state_dict.state_dict())
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    print("Applying PyTorch Dynamic INT8 Quantization...")
    # 3. Apply Dynamic Quantization
    # We target the heavy layers: LSTM and Linear
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )
    
    print(f"Saving quantized model to {output_path}...")
    torch.save(quantized_model.state_dict(), output_path)
    print("Quantization complete!")
    
    # 4. Compare file sizes
    fp32_size = os.path.getsize(model_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n--- Model Size Comparison ---")
    print(f"Original (FP32): {fp32_size:.2f} MB")
    print(f"Quantized (INT8): {int8_size:.2f} MB")
    print(f"Reduction Factor: {fp32_size / int8_size:.1f}x")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "15_07_model.pth")
    output_path = os.path.join(current_dir, "ev_pinn_int8.pth")
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}.")
    else:
        quantize_pytorch_model(input_path, output_path)
