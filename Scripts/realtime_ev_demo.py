import torch
import torch.nn as nn
import time
import os
import sys
import pickle
import numpy as np
from export_to_onnx import EVRangePINN

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_edge_model(model_path):
    model = EVRangePINN(input_size=16)
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.LSTM, nn.Linear}, 
        dtype=torch.qint8
    )
    quantized_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    quantized_model.eval()
    for param in quantized_model.parameters():
        param.requires_grad = False
    return quantized_model

def simulate_realtime_dashboard(model, data_path):
    print("Initializing Edge AI System...")
    print(f"Loading real empirical dataset from {os.path.basename(data_path)}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    X_test = data['X_test'] # Shape: (N, 10, 16)
    
    # Pick a random starting point in the test set to stream 100 continuous seconds
    start_idx = np.random.randint(0, len(X_test) - 100)
    
    time.sleep(1)
    print("\nStarting Real-time Inference Stream with Actual Traffic Data...")
    time.sleep(1)
    
    try:
        for step in range(100):
            # 1. Fetch real sensor data sliding window
            # Shape (1, 10, 16)
            input_tensor = torch.tensor(X_test[start_idx + step:start_idx + step + 1], dtype=torch.float32)
            
            # Extract real features from the most recent timestep in the window
            current_speed_kmh = input_tensor[0, -1, 0].item()
            elevation = input_tensor[0, -1, 1].item()
            ambient_temp = input_tensor[0, -1, 3].item()
            accel = input_tensor[0, -1, 4].item()
            
            # 2. Run Edge Inference
            t0 = time.time()
            with torch.no_grad():
                soc_pred, power_pred, range_pred, _ = model(input_tensor)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0
            
            # Extract scalar values
            soc_display = soc_pred.item() * 100.0
            pwr_val = power_pred.item()
            rng_val = range_pred.item()
            
            # 3. Render Dashboard
            clear_console()
            print("======================================================")
            print("           VEHICLE EDGE DASHBOARD (PINN INT8)         ")
            print("======================================================")
            print(f" TIME: {step:03d}s | EMPIRICAL DATASET STREAM")
            print("-" * 54)
            print(f" [SENSORS] Speed: {current_speed_kmh:5.1f} km/h  |  Accel: {accel:5.2f} m/s^2")
            print(f"           Temp : {ambient_temp:5.1f} °C     |  Grade: {elevation:5.1f} %")
            print("-" * 54)
            print(f" [AI PRED] Power Draw    : {pwr_val:6.2f} kW")
            print(f"           Predicted SOC : {soc_display:6.2f} %")
            print(f"           ESTIMATED RNG : {rng_val:6.1f} km")
            print("-" * 54)
            print(f" [SYS] Inference Latency : {latency_ms:5.2f} ms")
            print("======================================================")
            print("Press Ctrl+C to stop simulation...")
            
            # Wait to simulate 1Hz data rate
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    int8_path = os.path.join(current_dir, "ev_pinn_int8.pth")
    # The pkl is in the root project folder
    data_path = os.path.abspath(os.path.join(current_dir, "..", "processed_ev_data2 copy.pkl"))
    
    if not os.path.exists(int8_path):
        print("Error: INT8 model not found. Run quantize_model.py first.")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)
        
    torch.set_num_threads(2) # Edge constraint
    model = load_edge_model(int8_path)
    simulate_realtime_dashboard(model, data_path)
