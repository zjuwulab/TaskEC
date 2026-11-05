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

# ====== Hyperparameters (Optimized) ======
HIDDEN_DIM_MULTIPLIER = 2.0  
LATENT_DIM_MULTIPLIER = 1.0  
SIMULATION_LENGTH = 300     
NOISE_LEVEL = 0.1            
EPOCHS = 120                 
BATCH_SIZE = 50              
LR = 2e-3                    
L2 = 5e-5                    
ROI_NUM = 379
STEPS = 3                    
TR = 0.72

# ====== Condition Settings ======
GROUPA_KEYS = {'face': 'face', 'body': 'body', 'places': 'places', 'tools': 'tools'}
GROUPB_KEYS = {'2bk': '2bk', '0bk': '0bk'}

# ====== Directories ======
DATA_ROOT = "~/TaskEC/workmemory/timeseries"
OUTPUT_ROOT = "~/TaskEC/workmemory/step03_FinalModel"
SUBJECT_RANGE = "0"

# ====== Model Definition ======
class EnhancedANN_MLP(nn.Module):
    """Enhanced MLP with BatchNorm and Dropout for robust brain dynamics modeling."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = int(HIDDEN_DIM_MULTIPLIER * output_dim)
        latent_dim = int(LATENT_DIM_MULTIPLIER * output_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ====== Utility Functions ======
def multi2one(time_series, steps):
    """Convert time series to windowed input-target pairs."""
    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    input_X = np.zeros((n_step - steps, n_area * steps))
    target_Y = np.zeros((n_step - steps, n_area))
    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps+i].flatten()
        target_Y[i] = time_series[steps+i].flatten()
    return input_X, target_Y

def parse_evs(ev_file):
    """Parse HCP EV files (Condition, Onset, Duration)."""
    ev_data = []
    if not os.path.exists(ev_file): return ev_data
    with open(ev_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                ev_data.append([parts[0], float(parts[1]), float(parts[2])])
    return ev_data

def extract_condition_segments(signal, ev_data, tr):
    """Map task conditions to signal indices."""
    segments = {}
    for cond, onset, duration in ev_data:
        start_idx = int(round(onset / tr))
        end_idx = int(round((onset + duration) / tr))
        if cond not in segments: segments[cond] = []
        segments[cond].append((start_idx, end_idx))
    return segments

def collect_segments_as_signal(signal, segments):
    """Concatenate segments of a specific condition."""
    parts = []
    for start, end in segments:
        if start < signal.shape[0] and end <= signal.shape[0]:
            parts.append(signal[start:end])
    return np.vstack(parts) if parts else None

def compute_npi_ec(model, node_num, steps):
    """Infer Effective Connectivity via Virtual Neural Perturbation."""
    model.eval()
    ec_matrix = np.zeros((node_num, node_num))
    baseline_input = torch.zeros((1, node_num * steps)).to(device)
    with torch.no_grad():
        baseline_output = model(baseline_input).cpu().numpy()
        for i in range(node_num):
            perturbed_input = baseline_input.clone()
            # Perturb the most recent time point for region i
            perturbed_input[0, (steps-1)*node_num + i] += 1.0 
            perturbed_output = model(perturbed_input).cpu().numpy()
            ec_matrix[i, :] = perturbed_output - baseline_output
    return ec_matrix

def train_and_metrics(signal, steps):
    """Train the model on combined data and return evaluation metrics."""
    input_X, target_Y = multi2one(signal, steps)
    split_idx = int(0.8 * input_X.shape[0])
    
    train_x = torch.tensor(input_X[:split_idx], dtype=torch.float).to(device)
    train_y = torch.tensor(target_Y[:split_idx], dtype=torch.float).to(device)
    test_x = torch.tensor(input_X[split_idx:], dtype=torch.float).to(device)
    test_y = torch.tensor(target_Y[split_idx:], dtype=torch.float).to(device)

    model = EnhancedANN_MLP(ROI_NUM * steps, ROI_NUM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []
    loader = data.DataLoader(data.TensorDataset(train_x, train_y), BATCH_SIZE, shuffle=True)

    for _ in range(EPOCHS):
        model.train()
        for X, y in loader:
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_losses.append(criterion(model(train_x), train_y).item())
            test_losses.append(criterion(model(test_x), test_y).item())

    # Performance Evaluation
    model.eval()
    with torch.no_grad():
        pred_y = model(test_x).cpu().numpy()
        actual_y = test_y.cpu().numpy()
        r2 = r2_score(actual_y.flatten(), pred_y.flatten())
        mse = mean_squared_error(actual_y, pred_y)

    # FC Reconstruction
    sim_data = [np.zeros(ROI_NUM) for _ in range(steps)]
    with torch.no_grad():
        for _ in range(SIMULATION_LENGTH):
            noise = NOISE_LEVEL * np.random.randn(steps * ROI_NUM)
            inp = torch.tensor(np.array(sim_data[-steps:]).flatten() + noise, dtype=torch.float).to(device)
            sim_data.append(model(inp.unsqueeze(0)).cpu().numpy()[0])
    
    sim_fc = np.corrcoef(np.array(sim_data[steps:]).T)
    emp_fc = np.corrcoef(signal.T)
    fc_r, fc_p = pearsonr(sim_fc[np.triu_indices(ROI_NUM, 1)], emp_fc[np.triu_indices(ROI_NUM, 1)])
    
    # EC Inference
    ec_matrix = compute_npi_ec(model, ROI_NUM, steps)

    return {
        'model': model, 'train_loss': train_losses, 'test_loss': test_losses,
        'r2': r2, 'mse': mse, 'fc_r': fc_r, 'fc_p': fc_p,
        'empirical_FC': emp_fc, 'model_FC': sim_fc, 'NPI_EC': ec_matrix
    }

# ====== Main Processing per Subject ======
def process_subject(subj_id, subj_dir, output_root):
    print(f"\n=== Processing subject {subj_id} ===")
    subj_out_root = os.path.join(output_root, subj_id)
    os.makedirs(subj_out_root, exist_ok=True)

    run_files = sorted([f for f in os.listdir(subj_dir) if f.startswith("HCP_MMP")])
    ev_files = sorted([f for f in os.listdir(subj_dir) if f.startswith("EVs")])
    
    if len(run_files) < 2 or len(ev_files) < 2: return

    signals_runs = []
    for rf in run_files[:2]:
        sig = np.loadtxt(os.path.join(subj_dir, rf))
        sig = clean(sig, detrend=True, standardize='zscore_sample', high_pass=0.008, t_r=TR)
        signals_runs.append(sig)

    signals_all = np.vstack(signals_runs)
    evs_runs = [parse_evs(os.path.join(subj_dir, ef)) for ef in ev_files[:2]]
    segs_runs = [extract_condition_segments(signals_runs[i], evs_runs[i], TR) for i in range(2)]

    conditions_data = {}
    # Combine Group A & B conditions
    for group_keys in [GROUPA_KEYS, GROUPB_KEYS]:
        for key_word, outname in group_keys.items():
            parts = []
            for i in range(2):
                segs = [s for c, s in segs_runs[i].items() if key_word.lower() in c.lower()]
                flat_segs = [item for sublist in segs for item in sublist]
                if flat_segs:
                    p = collect_segments_as_signal(signals_runs[i], flat_segs)
                    if p is not None: parts.append(p)
            conditions_data[outname] = np.vstack(parts) if parts else None

    # Rest condition extraction
    task_mask = np.zeros(signals_all.shape[0], dtype=bool)
    offsets = [0, signals_runs[0].shape[0] * TR]
    for i, evs in enumerate(evs_runs):
        for _, onset, duration in evs:
            s = int(round((onset + offsets[i]) / TR))
            e = int(round((onset + duration + offsets[i]) / TR))
            task_mask[max(0, s):min(signals_all.shape[0], e)] = True
    conditions_data['rest'] = signals_all[~task_mask]

    # Training and Output
    for cond_name, cond_sig in conditions_data.items():
        if cond_sig is None: continue
        print(f"  Training: {cond_name} (Shape: {cond_sig.shape})")
        res = train_and_metrics(cond_sig, STEPS)
        
        cond_dir = os.path.join(subj_out_root, cond_name)
        os.makedirs(cond_dir, exist_ok=True)
        
        # Save results
        torch.save(res['model'].state_dict(), os.path.join(cond_dir, f"{subj_id}_{cond_name}_model.pth"))
        np.savetxt(os.path.join(cond_dir, f"{subj_id}_{cond_name}_NPI_EC.txt"), res['NPI_EC'])
        np.savetxt(os.path.join(cond_dir, f"{subj_id}_{cond_name}_empirical_FC.txt"), res['empirical_FC'])
        np.savetxt(os.path.join(cond_dir, f"{subj_id}_{cond_name}_model_FC.txt"), res['model_FC'])
        
        with open(os.path.join(cond_dir, f"{subj_id}_{cond_name}_metrics.txt"), 'w') as f:
            f.write(f"R2: {res['r2']:.4f}\nMSE: {res['mse']:.6f}\nFC_r: {res['fc_r']:.4f}")

def main():
    subs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    start, end = map(int, SUBJECT_RANGE.split(":"))
    for subj_id in subs[start-1:end]:
        try: process_subject(subj_id, os.path.join(DATA_ROOT, subj_id), OUTPUT_ROOT)
        except Exception as e: print(f"Error {subj_id}: {e}")

if __name__ == "__main__":
    main()