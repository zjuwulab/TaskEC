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

# ====== Model Definitions ======
class ANN_MLP(nn.Module):
    """Multilayer Perceptron as a proxy brain."""
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

class ANN_RNN(nn.Module):
    """Recurrent Neural Network as a proxy brain."""
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, data_length):
        super().__init__()
        self.input_dim = input_dim
        self.data_length = data_length
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(device)
        self.rnn = nn.RNN(input_size=1, hidden_size=latent_dim, batch_first=True).to(device)
        self.output = nn.Linear(latent_dim, output_dim).to(device)

    def forward(self, x):
        x = x.view(-1, self.data_length, self.input_dim)
        encodes = self.enc(x)
        ht, _ = self.rnn(torch.zeros((x.shape[0], 1, 1)).to(device), 
                        torch.permute(encodes[:, self.data_length-1:, :], (1, 0, 2)).contiguous())
        return self.output(ht)[:,0,:]

class ANN_VAR(nn.Module):
    """Vector Autoregression as a proxy brain."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.func = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, x):
        return self.func(x)

class ANN_LSTM(nn.Module):
    """Long Short-Term Memory network as a proxy brain."""
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, data_length, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.data_length = data_length
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ).to(device)
        self.lstm = nn.LSTM(input_size=1, hidden_size=latent_dim, 
                           batch_first=True, num_layers=num_layers).to(device)
        self.output = nn.Linear(latent_dim, output_dim).to(device)

    def forward(self, x):
        x = x.view(-1, self.data_length, self.input_dim)
        encodes = self.enc(x)
        ht, _ = self.lstm(torch.zeros((x.shape[0], 1, 1)).to(device), 
                         (torch.permute(encodes[:, self.data_length-1:, :], (1, 0, 2)).contiguous(),
                          torch.zeros_like(torch.permute(encodes[:, self.data_length-1:, :], (1, 0, 2)).contiguous())))
        return self.output(ht)[:,0,:]

class ANN_Residual(nn.Module):
    """MLP with Residual Connections as a proxy brain."""
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, num_blocks=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim).to(device)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ).to(device) for _ in range(num_blocks)
        ])
        self.output_fc = nn.Linear(hidden_dim, output_dim).to(device)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.input_fc(x))
        for block in self.blocks:
            residual = x
            x = self.relu(block(x) + residual)
            x = self.dropout(x)
        return self.output_fc(x)

# ====== Functions ======
def multi2one(time_series, steps):
    """Splits time series into input-target pairs (sliding window)."""
    n_area = time_series.shape[1]
    n_step = time_series.shape[0]
    input_X = np.zeros((n_step - steps, n_area * steps))
    target_Y = np.zeros((n_step - steps, n_area))
    for i in range(n_step - steps):
        input_X[i] = time_series[i:steps+i].flatten()
        target_Y[i] = time_series[steps+i].flatten()
    return input_X, target_Y

def train_NN(model, input_X, target_Y, batch_size=50, train_set_proportion=0.8, num_epochs=100, lr=1e-3, l2=0):
    """Trains the neural network model using empirical data."""
    split_idx = int(train_set_proportion * input_X.shape[0])
    train_inputs = torch.tensor(input_X[:split_idx], dtype=torch.float).to(device)
    train_targets = torch.tensor(target_Y[:split_idx], dtype=torch.float).to(device)
    test_inputs = torch.tensor(input_X[split_idx:], dtype=torch.float).to(device)
    test_targets = torch.tensor(target_Y[split_idx:], dtype=torch.float).to(device)
    
    train_iter = data.DataLoader(data.TensorDataset(train_inputs, train_targets), batch_size, shuffle=True)
    test_iter = data.DataLoader(data.TensorDataset(test_inputs, test_targets), batch_size, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    train_epoch_loss, test_epoch_loss = [], []
    for _ in range(num_epochs):
        model.train()
        for X, y in train_iter:
            y_hat = model(X)
            l = loss_fn(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for it, storage in [(train_iter, train_epoch_loss), (test_iter, test_epoch_loss)]:
                total_l, total_n = 0, 0
                for X, y in it:
                    y_hat = model(X)
                    total_l += loss_fn(y_hat, y) * y.shape[0]
                    total_n += y.shape[0]
                storage.append(float(total_l / total_n))
    
    return model, train_epoch_loss, test_epoch_loss

def corrcoef(signals):
    """Calculates Functional Connectivity (FC) matrix."""
    return torch.corrcoef(torch.tensor(signals.T, dtype=torch.float).to(device)).detach().cpu().numpy()

def model_FC(model, node_num, steps):
    """Simulates data with random noise to calculate Model FC."""
    sim_data = [np.zeros(node_num) for _ in range(steps)]
    for _ in range(1200):
        noise = 0.1 * np.random.randn(steps * node_num)
        model_input = torch.tensor(np.array(sim_data[-steps:]).flatten() + noise, dtype=torch.float).to(device)
        pred = model(model_input).detach().cpu().numpy()
        sim_data.append(pred[0] if isinstance(model, (ANN_RNN, ANN_LSTM)) else pred)
    return corrcoef(np.array(sim_data))

def model_EC(model, input_X, target_Y, pert_strength):
    """Infers Effective Connectivity (EC) via perturbation."""
    node_num = target_Y.shape[1]
    steps = int(input_X.shape[1] / node_num)
    NPI_EC = np.zeros((node_num, node_num))
    input_tensor = torch.tensor(input_X, dtype=torch.float).to(device)
    unperturbed_output = model(input_tensor).detach().cpu().numpy()
    
    for node in range(node_num):
        perturbation = np.zeros((steps, node_num))
        perturbation[-1, node] = pert_strength
        perturbed_input = torch.tensor(input_X + perturbation.flatten(), dtype=torch.float).to(device)
        perturbed_output = model(perturbed_input).detach().cpu().numpy()
        NPI_EC[node] = np.mean(perturbed_output - unperturbed_output, axis=0)
    return NPI_EC

# ====== Processing ======
def create_model(model_type, input_dim, output_dim, steps):
    """Factory function for creating models."""
    p = MODEL_PARAMS[model_type]
    if model_type == 'MLP': return ANN_MLP(input_dim, p['hidden_dim'], p['latent_dim'], output_dim)
    if model_type == 'CNN': return ANN_CNN(output_dim, p['hidden_dim'], p['latent_dim'], steps)
    if model_type == 'RNN': return ANN_RNN(output_dim, p['hidden_dim'], p['latent_dim'], output_dim, steps)
    if model_type == 'VAR': return ANN_VAR(input_dim, output_dim)
    if model_type == 'LSTM': return ANN_LSTM(output_dim, p['hidden_dim'], p['latent_dim'], output_dim, steps)
    if model_type == 'Residual': return ANN_Residual(input_dim, p['hidden_dim'], p['latent_dim'], output_dim)
    raise ValueError(f"Unknown model type: {model_type}")

# ====== Parameters ======
TR = 0.72
DATA_ROOT = "~/TaskEC/workmemory/timeseries"
OUTPUT_ROOT = "~/TaskEC/workmemory/step01/results/sSsTsC_step1"
SUBJECT_RANGE = "1:100"
ROI_NUM, STEPS, EPOCHS, LR, L2, MIN_SAMPLES = 379, 3, 100, 1e-4, 1e-4, 10
SELECTED_MODELS = ['MLP', 'RNN', 'VAR', 'LSTM', 'Residual']

MODEL_PARAMS = {
    'MLP': {'hidden_dim': 2*ROI_NUM, 'latent_dim': int(0.8*ROI_NUM)},
    'RNN': {'hidden_dim': 2*ROI_NUM, 'latent_dim': int(0.8*ROI_NUM)},
    'VAR': {},
    'LSTM': {'hidden_dim': 2*ROI_NUM, 'latent_dim': int(0.8*ROI_NUM)},
    'Residual': {'hidden_dim': 2*ROI_NUM, 'latent_dim': int(0.8*ROI_NUM), 'num_blocks': 2}
}

GROUPA_KEYS = {'faces': 'face', 'body': 'body', 'places': 'places', 'tools': 'tools'}
GROUPB_KEYS = {'2bk': '2bk', '0bk': '0bk'}

def get_subject_list(root_dir, subj_range):
    subs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if isinstance(subj_range, str) and ":" in subj_range:
        start, end = map(int, subj_range.split(":"))
        return subs[start-1:end]
    return subs[:int(subj_range)] if str(subj_range).isdigit() and int(subj_range) != 0 else subs

def parse_evs(ev_file):
    data = np.genfromtxt(ev_file, dtype=str, skip_header=1)
    return data[np.newaxis, :] if data.ndim == 1 else data

def extract_condition_segments(signals, ev_data, tr):
    """Map EV timing to signal indices."""
    segs = {}
    for cond_name, onset, duration in ev_data:
        start = max(0, int(round(float(onset) / tr)))
        end = min(signals.shape[0], int(round((float(onset) + float(duration)) / tr)))
        if end > start: segs.setdefault(cond_name, []).append((start, end))
    return segs

def collect_segments(signals, seg_list):
    return np.vstack([signals[s:e] for s, e in seg_list]) if seg_list else None

def flatten_no_diag(mat):
    return mat[~np.eye(mat.shape[0], dtype=bool)]

def train_and_eval(inputs, targets, model_type):
    """Common training and metric calculation wrapper."""
    if inputs.shape[0] < MIN_SAMPLES: return None
    model = create_model(model_type, inputs.shape[1], targets.shape[1], STEPS)
    model, t_loss, v_loss = train_NN(model, inputs, targets, num_epochs=EPOCHS, lr=LR, l2=L2)
    
    n_train = int(0.8 * inputs.shape[0])
    with torch.no_grad():
        y_pred = model(torch.tensor(inputs[n_train:], dtype=torch.float).to(device)).cpu().numpy()
    
    mse = mean_squared_error(targets[n_train:], y_pred)
    r2 = r2_score(targets[n_train:], y_pred)
    emp_fc = corrcoef(targets)
    m_fc = model_FC(model, ROI_NUM, STEPS)
    ec = model_EC(model, inputs, targets, 1.0)
    np.fill_diagonal(ec, 0)
    
    r, p = pearsonr(flatten_no_diag(m_fc), flatten_no_diag(emp_fc))
    return {'model': model, 'train_loss': t_loss, 'test_loss': v_loss, 'mse': mse, 'r2': r2, 
            'empirical_FC': emp_fc, 'model_FC': m_fc, 'NPI_EC': ec, 'fc_r': r, 'fc_p': p}

def process_subject(subj_id, subj_dir, output_root):
    print(f"--- Processing {subj_id} ---")
    subj_out = os.path.join(output_root, subj_id)
    os.makedirs(subj_out, exist_ok=True)

    run_files = sorted([f for f in os.listdir(subj_dir) if f.startswith("HCP_MMP")])
    ev_files = sorted([f for f in os.listdir(subj_dir) if f.startswith("EVs")])
    if not run_files or not ev_files: return

    signals_runs = [clean(np.loadtxt(os.path.join(subj_dir, f)), detrend=True, standardize='zscore_sample', high_pass=0.008, t_r=TR) for f in run_files]
    evs_runs = [parse_evs(os.path.join(subj_dir, f)) for f in ev_files]
    segs_runs = [extract_condition_segments(sig, evs, TR) for sig, evs in zip(signals_runs, evs_runs)]

    conditions = {}
    # Extract Task Conditions
    for group in [GROUPA_KEYS, GROUPB_KEYS]:
        for key, outname in group.items():
            parts = [collect_segments(sig, [s for c, sl in segs.items() if key.lower() in c.lower() for s in sl]) 
                     for sig, segs in zip(signals_runs, segs_runs)]
            valid_parts = [p for p in parts if p is not None]
            conditions[outname] = {'combined': np.vstack(valid_parts) if valid_parts else None, 'per_run': parts}

    # Extract Rest
    all_sig = np.vstack(signals_runs)
    mask = np.zeros(all_sig.shape[0], dtype=bool)
    offset = 0
    for evs in evs_runs:
        for _, onset, dur in evs:
            s, e = int(round(float(onset)/TR)), int(round((float(onset)+float(dur))/TR))
            mask[offset+s:offset+e] = True
        offset += signals_runs[0].shape[0] # assuming equal run lengths
    conditions['rest'] = {'combined': all_sig[~mask], 'per_run': [None, None]}

    # Train and Save
    for cond_name, info in conditions.items():
        if info['combined'] is None: continue
        for m_type in SELECTED_MODELS:
            m_dir = os.path.join(subj_out, cond_name, m_type)
            os.makedirs(m_dir, exist_ok=True)
            
            # 1. Combined Run training
            inp, tar = multi2one(info['combined'], STEPS)
            res = train_and_eval(inp, tar, m_type)
            if res:
                torch.save(res['model'].state_dict(), os.path.join(m_dir, f"model.pth"))
                np.savetxt(os.path.join(m_dir, "NPI_EC.txt"), res['NPI_EC'])
                # Plot Loss
                plt.figure()
                plt.plot(res['train_loss'], label='Train')
                plt.plot(res['test_loss'], label='Test')
                plt.legend(); plt.savefig(os.path.join(m_dir, "loss.png")); plt.close()

def main():
    subjects = get_subject_list(DATA_ROOT, SUBJECT_RANGE)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for sub in subjects:
        try:
            process_subject(sub, os.path.join(DATA_ROOT, sub), OUTPUT_ROOT)
        except Exception as e:
            print(f"Error {sub}: {e}")

if __name__ == "__main__":
    main()