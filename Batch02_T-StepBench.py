import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from nilearn.signal import clean

# ====== Device Configuration ======
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ====== Model Definition ======
class ANN_MLP(nn.Module):
    """Multilayer Perceptron for brain dynamics modeling."""
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        ).to(device)

    def forward(self, x):
        return self.func(x)

# ====== Core Utility Functions ======
def multi2one(time_series, steps):
    """Convert time series to supervised learning format (X: past steps, Y: current)."""
    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    input_X = np.zeros((n_step - steps, n_area * steps))
    target_Y = np.zeros((n_step - steps, n_area))
    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps+i].flatten()
        target_Y[i] = time_series[steps+i].flatten()
    return input_X, target_Y

def train_MLP(model, input_X, target_Y, batch_size=50, train_ratio=0.8, num_epochs=100, lr=1e-3, l2=0):
    """Train the MLP model using brain signals."""
    split_idx = int(train_ratio * input_X.shape[0])
    train_inputs = torch.tensor(input_X[:split_idx], dtype=torch.float).to(device)
    train_targets = torch.tensor(target_Y[:split_idx], dtype=torch.float).to(device)
    test_inputs = torch.tensor(input_X[split_idx:], dtype=torch.float).to(device)
    test_targets = torch.tensor(target_Y[split_idx:], dtype=torch.float).to(device)
    
    train_loader = data.DataLoader(data.TensorDataset(train_inputs, train_targets), batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    for _ in range(num_epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
        
        # Performance logging per epoch
        model.eval()
        with torch.no_grad():
            train_losses.append(float(criterion(model(train_inputs), train_targets)))
            test_losses.append(float(criterion(model(test_inputs), test_targets)))
    
    return model, train_losses, test_losses

def calculate_fc(signals):
    """Compute Functional Connectivity (Pearson correlation matrix)."""
    return torch.corrcoef(torch.tensor(signals.T, dtype=torch.float).to(device)).cpu().numpy()

# ====== Main Processing Parameters ======
TR = 0.72
DATA_ROOT = "~/TaskEC/workmemory/timeseries"
OUTPUT_ROOT = "~/TaskEC/workmemory/step02_TStepBench"
SUBJECT_RANGE = "1:100"  # Process first 100 subjects
ROI_NUM = 379
STEPS_LIST = [1, 2, 3, 4, 5]  # Comparing different look-back windows
EPOCHS, LR, L2 = 100, 1e-4, 1e-4

# Condition Mapping
TASK_CONDITIONS = {'face': 'face', 'body': 'body', '2bk': '2bk', '0bk': '0bk', 'rest': 'rest'}

def process_subject_steps(subj_id, subj_dir, output_root):
    """Loop through task conditions and test multiple time-steps for MLP."""
    print(f"--- Processing {subj_id} ---")
    
    # Load and Preprocess BOLD (Simplified logic)
    run_files = sorted([f for f in os.listdir(subj_dir) if f.startswith("HCP_MMP")])
    if not run_files: return
    
    # Placeholder for signal loading (assuming signals are pre-extracted/cleaned)
    # Combined training on all available task data for this benchmarking phase
    for steps in STEPS_LIST:
        print(f" Testing Steps: {steps}")
        for cond_name in TASK_CONDITIONS.values():
            # [Logic to extract specific condition segments based on EV files goes here]
            # For brevity, assuming 'combined_signal' is obtained for the condition
            # combined_signal = extract_logic(subj_dir, cond_name)
            
            # --- Training and Evaluation ---
            # input_X, target_Y = multi2one(combined_signal, steps)
            # model = ANN_MLP(ROI_NUM * steps, 2*ROI_NUM, int(0.8*ROI_NUM), ROI_NUM)
            # trained_model, t_loss, v_loss = train_MLP(model, input_X, target_Y, num_epochs=EPOCHS, lr=LR, l2=L2)
            
            # Save metrics (MSE, R2, FC-correlation) to output_root/subj_id/cond/steps/
            pass

if __name__ == "__main__":
    # Main execution logic remains similar to your original script
    # but focuses strictly on comparing STEPS_LIST.
    pass