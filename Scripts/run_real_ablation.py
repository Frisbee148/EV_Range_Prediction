import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

# ──────────────────────────────────────────────
# 1. Models Definition (layer names match 15_07_model.pth)
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
        self.register_buffer('regen_efficiency', torch.tensor(0.7))
        self.register_buffer('aux_power', torch.tensor(1.5))
        self.register_buffer('temp_ref', torch.tensor(25.0))
        self.register_buffer('temp_coeff', torch.tensor(0.005))
        self.register_buffer('min_energy_per_km', torch.tensor(0.15))
        self.register_buffer('max_energy_per_km', torch.tensor(0.50))

    def calculate_theoretical_power(self, velocity_kmh, acceleration_ms2, elevation_grade, ambient_temp):
        v  = torch.clamp(velocity_kmh / 3.6, min=0.01)
        Fd = 0.5 * self.air_density * self.drag_coeff * self.frontal_area * v**2
        Fr = self.rolling_resistance * self.vehicle_mass * self.gravity
        Fg = self.vehicle_mass * self.gravity * torch.sin(torch.atan(torch.clamp(elevation_grade / 100.0, -0.3, 0.3)))
        Fa = self.vehicle_mass * torch.clamp(acceleration_ms2, -5.0, 5.0)
        wheel_power = (Fd + Fr + Fg + Fa) * v / 1000.0
        tf = 1.0 - self.temp_coeff * torch.abs(ambient_temp - self.temp_ref)
        eff = self.drivetrain_efficiency * torch.clamp(tf, 0.7, 1.0)
        motor = torch.where(wheel_power >= 0, wheel_power / eff, wheel_power * self.regen_efficiency)
        return torch.clamp(motor + self.aux_power, -50.0, 100.0)

    def forward(self, predictions, inputs):
        v   = torch.clamp(inputs[:, -1, 0], min=0.1)
        el  = inputs[:, -1, 1]
        acc = inputs[:, -1, 4]
        tmp = inputs[:, -1, 3]
        phys = self.calculate_theoretical_power(v, acc, el, tmp)
        pred_p = predictions['power'].squeeze()
        power_loss = F.huber_loss(pred_p, phys, delta=5.0)
        soc_pred   = predictions['soc'].squeeze()
        soc_loss   = torch.mean(F.relu(soc_pred - 1.0) + F.relu(-soc_pred))
        mag_loss   = torch.mean(F.relu(torch.abs(pred_p) - 80.0))
        total = power_loss + 0.5 * soc_loss + 0.1 * mag_loss
        return {
            'power_physics': power_loss, 'soc_constraint': soc_loss,
            'power_magnitude': mag_loss, 'total_physics': total
        }

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        return torch.sum(weights * lstm_output, dim=1)

# ── Full Model: layer names match 15_07_model.pth exactly ──
class EVRangePINN_Full(nn.Module):
    """Full Model with PINN and Attention — architecture matches saved checkpoint."""
    def __init__(self, input_size=16, hidden_size=256, num_layers=3, dropout=0.3):
        super(EVRangePINN_Full, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.physics_layer = PhysicsLayer()
        self.feature_extractor = nn.ModuleDict({
            'soc_features': nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.ReLU(), nn.Dropout(dropout)),
            'power_features': nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.ReLU(), nn.Dropout(dropout))
        })
        self.soc_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout // 2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.power_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout // 2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        # range_estimator exists in checkpoint but we don't need it for ablation eval
        # Skipped — strict=False will ignore those keys during loading

    def forward(self, x):
        x_proj = self.input_projection(x)
        lstm_out, _ = self.lstm(x_proj)
        features = self.attention(lstm_out)
        soc_features = self.feature_extractor['soc_features'](features)
        power_features = self.feature_extractor['power_features'](features)
        soc_pred = self.soc_predictor(soc_features)
        power_pred = torch.tanh(self.power_predictor(power_features)) * 50.0
        physics_losses = self.physics_layer({'soc': soc_pred, 'power': power_pred}, x)
        return {'soc': soc_pred, 'power': power_pred, 'physics_losses': physics_losses}

# ── Ablated Model: No Attention — trains from scratch ──
class EVRangePINN_NoAttention(nn.Module):
    """Ablated Model: Removed Attention Layer"""
    def __init__(self, input_size=16, hidden_size=256, num_layers=3, dropout=0.3):
        super(EVRangePINN_NoAttention, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        # NO ATTENTION LAYER
        self.physics_layer = PhysicsLayer()
        self.feature_extractor = nn.ModuleDict({
            'soc_features': nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.ReLU(), nn.Dropout(dropout)),
            'power_features': nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.ReLU(), nn.Dropout(dropout))
        })
        self.soc_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout // 2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.power_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout // 2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x_proj = self.input_projection(x)
        lstm_out, _ = self.lstm(x_proj)
        # Use last hidden state instead of attention
        features = lstm_out[:, -1, :]
        soc_features = self.feature_extractor['soc_features'](features)
        power_features = self.feature_extractor['power_features'](features)
        soc_pred = self.soc_predictor(soc_features)
        power_pred = torch.tanh(self.power_predictor(power_features)) * 50.0
        physics_losses = self.physics_layer({'soc': soc_pred, 'power': power_pred}, x)
        return {'soc': soc_pred, 'power': power_pred, 'physics_losses': physics_losses}

# ──────────────────────────────────────────────
# 2. Training & Evaluation
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='none')
        return (self.alpha * (1 - torch.exp(-mse)) ** self.gamma * mse).mean()

def train_model(model, train_loader, device, epochs=15):
    """Train a model from scratch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    focal = FocalLoss()
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = (2.0 * focal(out['soc'], by[:, 0:1])
                    + F.huber_loss(out['power'], by[:, 1:2], delta=5.0)
                    + 0.3 * out['physics_losses']['total_physics'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {ep+1}/{epochs} — Loss: {epoch_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    """Evaluate and return Power MAE."""
    model.eval()
    pwr_t, pwr_p = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            pwr_t.extend(by[:, 1].cpu().numpy())
            pwr_p.extend(out['power'].squeeze().cpu().numpy())
    return mean_absolute_error(pwr_t, pwr_p)

# ──────────────────────────────────────────────
# 3. Main Runner
# ──────────────────────────────────────────────
if __name__ == '__main__':
    DATA_PATH = r"G:\My Drive\ev range prediction project agni\new data\measurable data\cleaned_data\cleaned final data\sumo\sumo_integration\preprocess_cleaned\Trip B\processed_ev_data2.pkl"
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "15_07_model.pth")

    print("Loading data...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    y_train = data['y_train']
    soc_min, soc_max = y_train[:, 0].min(), y_train[:, 0].max()
    y_train[:, 0] = (y_train[:, 0] - soc_min) / (soc_max - soc_min + 1e-8)

    y_test = data['y_test']
    y_test[:, 0] = (y_test[:, 0] - soc_min) / (soc_max - soc_min + 1e-8)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(data['X_train']), torch.FloatTensor(y_train)), batch_size=128, shuffle=True)
    test_loader  = DataLoader(TensorDataset(torch.FloatTensor(data['X_test']),  torch.FloatTensor(y_test)),  batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = data['X_train'].shape[2]
    print(f"Using device: {device}")
    print(f"Input size: {input_size}")

    # ── FULL MODEL: Load pre-trained weights (no retraining!) ──
    print("\n" + "="*50)
    print("FULL MODEL (With Attention) — Loading pre-trained weights")
    print("="*50)
    full_model = EVRangePINN_Full(input_size=input_size).to(device)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    # strict=False to skip range_estimator if shapes mismatch, etc.
    loaded = full_model.load_state_dict(checkpoint, strict=False)
    print(f"  Loaded checkpoint: {WEIGHTS_PATH}")
    if loaded.missing_keys:
        print(f"  Missing keys (OK if only range_estimator): {loaded.missing_keys}")
    if loaded.unexpected_keys:
        print(f"  Unexpected keys (ignored): {loaded.unexpected_keys}")
    full_mae = evaluate_model(full_model, test_loader, device)
    print(f"  Full Model Power MAE: {full_mae:.4f}")

    # ── ABLATED MODEL: Train from scratch ──
    print("\n" + "="*50)
    print("ABLATED MODEL (No Attention) — Training from scratch")
    print("="*50)
    no_attn_model = EVRangePINN_NoAttention(input_size=input_size).to(device)
    train_model(no_attn_model, train_loader, device, epochs=15)
    no_attn_mae = evaluate_model(no_attn_model, test_loader, device)
    print(f"  No-Attention Model Power MAE: {no_attn_mae:.4f}")

    # ── Results ──
    increase = ((no_attn_mae - full_mae) / full_mae) * 100
    print("\n" + "="*60)
    print("REAL ABLATION RESULTS")
    print("="*60)
    print(f"  Full Model (LSTM+Attn+PINN):   {full_mae:.4f} kW")
    print(f"  No Attention (LSTM+PINN):      {no_attn_mae:.4f} kW")
    print(f"  Performance Degradation:       +{increase:.1f}%")
    print("="*60)

    # Save results
    with open("real_ablation_results.txt", "w") as f:
        f.write("Real Ablation Study Results\n")
        f.write("="*60 + "\n")
        f.write(f"Full Model (LSTM+Attn+PINN):   {full_mae:.4f} kW\n")
        f.write(f"No Attention (LSTM+PINN):      {no_attn_mae:.4f} kW\n")
        f.write(f"Performance Degradation:       +{increase:.1f}%\n")
        f.write("="*60 + "\n")
    print("\nResults saved to real_ablation_results.txt")
