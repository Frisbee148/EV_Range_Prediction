import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ──────────────────────────────────────────────
# Correct Model Architecture (from extracted_code.py)
# ──────────────────────────────────────────────

class PhysicsLayer(nn.Module):
    def __init__(self):
        super(PhysicsLayer, self).__init__()
        self.register_buffer('battery_capacity', torch.tensor(42.2))
        self.register_buffer('vehicle_mass', torch.tensor(1300.0))
        self.register_buffer('drag_coeff', torch.tensor(0.29))
        self.register_buffer('frontal_area', torch.tensor(2.38))
        self.register_buffer('air_density', torch.tensor(1.225))
        self.register_buffer('rolling_resistance', torch.tensor(0.01))
        self.register_buffer('gravity', torch.tensor(9.81))
        
        self.register_buffer('drivetrain_efficiency', torch.tensor(0.85))
        self.register_buffer('regen_efficiency', torch.tensor(0.70))
        self.register_buffer('aux_power', torch.tensor(1.5))
        
        self.register_buffer('temp_ref', torch.tensor(25.0))
        self.register_buffer('temp_coeff', torch.tensor(0.005))
        
        self.register_buffer('min_energy_per_km', torch.tensor(0.12))
        self.register_buffer('max_energy_per_km', torch.tensor(0.50))
        
    def calculate_driving_forces(self, velocity_kmh, acceleration_ms2, elevation_grade):
        velocity_ms = torch.clamp(velocity_kmh / 3.6, min=0.01)
        drag_force = 0.5 * self.air_density * self.drag_coeff * self.frontal_area * (velocity_ms ** 2)
        rolling_force = self.rolling_resistance * self.vehicle_mass * self.gravity
        grade_rad = torch.atan(torch.clamp(elevation_grade / 100.0, min=-0.3, max=0.3))
        gravity_force = self.vehicle_mass * self.gravity * torch.sin(grade_rad)
        accel_force = self.vehicle_mass * torch.clamp(acceleration_ms2, min=-5.0, max=5.0)
        
        return {
            'drag_force': drag_force,
            'rolling_force': rolling_force,
            'gravity_force': gravity_force,
            'accel_force': accel_force,
            'total_force': drag_force + rolling_force + gravity_force + accel_force,
            'velocity_ms': velocity_ms
        }
    
    def calculate_theoretical_power(self, velocity_kmh, acceleration_ms2, elevation_grade, ambient_temp):
        forces = self.calculate_driving_forces(velocity_kmh, acceleration_ms2, elevation_grade)
        wheel_power = forces['total_force'] * forces['velocity_ms'] / 1000.0

        temp_factor = 1.0 - self.temp_coeff * torch.abs(ambient_temp - self.temp_ref)
        adjusted_efficiency = self.drivetrain_efficiency * torch.clamp(temp_factor, min=0.7, max=1.0)

        motor_power = torch.where(
            wheel_power >= 0,
            wheel_power / adjusted_efficiency,
            wheel_power * self.regen_efficiency
        )
        total_power = motor_power + self.aux_power
        return torch.clamp(total_power, min=-50.0, max=100.0)
    
    def forward(self, predictions, inputs):
        velocity = torch.clamp(inputs[:, -1, 0], min=0.1)
        elevation = inputs[:, -1, 1]
        acceleration = inputs[:, -1, 4]
        ambient_temp = inputs[:, -1, 3]
        
        physics_power = self.calculate_theoretical_power(velocity, acceleration, elevation, ambient_temp)
        
        predicted_power = predictions['power'].squeeze()
        power_loss = F.huber_loss(predicted_power, physics_power, delta=5.0)
        
        soc_pred = predictions['soc'].squeeze()
        soc_bound_loss = torch.mean(
            F.relu(soc_pred - 1.0) + F.relu(-soc_pred)
        )
        
        power_magnitude_loss = torch.mean(F.relu(torch.abs(predicted_power) - 80.0))
        
        return {
            'power_physics': power_loss,
            'soc_constraint': soc_bound_loss,
            'power_magnitude': power_magnitude_loss,
            'total_physics': power_loss + 0.5 * soc_bound_loss + 0.1 * power_magnitude_loss
        }

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(weights * lstm_output, dim=1)
        return attended_output

class EVRangePINN(nn.Module):
    def __init__(self, input_size=16, hidden_size=256, num_layers=3, dropout=0.3, physics_weight=0.5):
        super(EVRangePINN, self).__init__()
        self.physics_weight = physics_weight
        self.input_size = input_size
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, 
            dropout=dropout, batch_first=True, bidirectional=False
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.physics_layer = PhysicsLayer()
        
        self.feature_extractor = nn.ModuleDict({
            'soc_features': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'power_features': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        })
        
        self.soc_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.power_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.range_estimator = nn.Sequential(
            nn.Linear(hidden_size + 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
    def calculate_range(self, soc_pred, power_pred, velocity):
        remaining_energy = soc_pred * self.physics_layer.battery_capacity
        
        power_consumption = torch.clamp(torch.abs(power_pred), min=1.0, max=80.0)
        velocity_safe = torch.clamp(velocity, min=1.0)
        
        energy_rate = torch.clamp(power_consumption / velocity_safe, min=0.10, max=0.60)
        range_pred = torch.clamp(remaining_energy / energy_rate, min=0.0, max=800.0)
        
        return range_pred
        
    def forward(self, x):
        x_proj = self.input_projection(x)
        lstm_out, _ = self.lstm(x_proj)
        attended_features = self.attention(lstm_out)
        
        soc_features = self.feature_extractor['soc_features'](attended_features)
        power_features = self.feature_extractor['power_features'](attended_features)
        
        soc_pred = self.soc_predictor(soc_features)
        power_raw = self.power_predictor(power_features)
        power_pred = torch.tanh(power_raw) * 50.0
        
        velocity = x[:, -1, 0]
        range_pred = self.calculate_range(soc_pred.squeeze(), power_pred.squeeze(), velocity)
        
        range_features = torch.cat([attended_features, soc_pred, power_pred], dim=1)
        range_nn = self.range_estimator(range_features)
        
        final_range = 0.7 * range_pred.unsqueeze(1) + 0.3 * range_nn
        
        physics_losses = self.physics_layer({'soc': soc_pred, 'power': power_pred}, x)
        
        # Return tuple instead of dict for ONNX export
        return soc_pred, power_pred, final_range, physics_losses['total_physics']

# ──────────────────────────────────────────────
# Export Script
# ──────────────────────────────────────────────

def export_model_to_onnx(model_path, output_path):
    print(f"Loading PyTorch model from: {model_path}")
    
    # Initialize the model
    model = EVRangePINN(input_size=16)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(state_dict, nn.Module):
        model.load_state_dict(state_dict.state_dict())
    else:
        model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")
    
    model.eval()

    # Create dummy input: (batch_size=1, sequence_length=10, features=16)
    dummy_input = torch.randn(1, 10, 16)

    print(f"Exporting to ONNX format: {output_path}...")
    
    # Export to ONNX
    torch.onnx.export(
        model,                         # model being run
        dummy_input,                   # model input (or a tuple for multiple inputs)
        output_path,                   # where to save the model (can be a file or file-like object)
        export_params=True,            # store the trained parameter weights inside the model file
        opset_version=14,              # the ONNX version to export the model to
        do_constant_folding=True,      # whether to execute constant folding for optimization
        input_names = ['input'],       # the model's input names
        output_names = ['soc', 'power', 'range', 'physics_loss'], # the model's output names
        dynamic_axes={                 # variable length axes
            'input' : {0 : 'batch_size'},
            'soc' : {0 : 'batch_size'},
            'power' : {0 : 'batch_size'},
            'range': {0: 'batch_size'},
            'physics_loss' : {0 : 'batch_size'}
        }
    )
    
    print("Export complete!")
    
    # Check model size
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ONNX Model saved at {output_path} (Size: {size_mb:.2f} MB)")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "15_07_model.pth")
    output_path = os.path.join(current_dir, "ev_pinn_fp32.onnx")
    
    export_model_to_onnx(model_path, output_path)
