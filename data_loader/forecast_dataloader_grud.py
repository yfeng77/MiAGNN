import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd


# -----------------------------------------------------------------------------
# Normalization utilities
# -----------------------------------------------------------------------------

def normalized(data: np.ndarray, normalize_method: str, norm_statistic=None):
    """Normalize input data using either min‑max or z‑score statistics."""
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min']) + 1e-5
        data = (data - np.array(norm_statistic['min'])) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = np.array(norm_statistic['mean'])
        std = np.array(norm_statistic['std'])
        std = np.where(std == 0, 1, std)  # avoid divide‑by‑zero
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data: np.ndarray, normalize_method: str, norm_statistic):
    """Inverse transform for normalized data."""
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min']) + 1e-8
        data = data * scale + np.array(norm_statistic['min'])
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = np.array(norm_statistic['mean'])
        std = np.array(norm_statistic['std'])
        std = np.where(std == 0, 1, std)
        data = data * std + mean
    return data


def de_normalized_batch(data: np.ndarray, normalize_method: str, para1: np.ndarray, para2: np.ndarray):
    """Inverse transform applied per‑batch for 3D arrays.

    Parameters
    ----------
    data : np.ndarray, shape (B, H, N)
    para1, para2 : np.ndarray
        Batch‑level normalization parameters (mean/std or min/max).
    """
    if normalize_method == 'z_score':
        stds = np.where(para2 == 0, 1, para2)
        for step in range(data.shape[1]):
            data[:, step, :] = data[:, step, :] * stds + para1
        return data
    elif normalize_method == 'min_max':
        scale = para2 - para1 + 1e-8
        for step in range(data.shape[1]):
            data[:, step, :] = data[:, step, :] * scale + para1
        return data


# -----------------------------------------------------------------------------
# GRU‑D preprocessing
# -----------------------------------------------------------------------------

def grudData(data: np.ndarray, granularity: int):
    """Compute GRU‑D input augmentation: mask M and delta time Δ.

    For each variable (column), mark observed entries with mask = 1.
    For missing values, impute 0 and increment Δ by the time since last observation.

    Returns
    -------
    new_X : np.ndarray, shape (T, N, 3)
        Concatenated array of [X, M, Δ].
    """
    X = data.copy()
    num_time_steps, num_variables = X.shape

    # Create timestamp series S and initialize matrices
    S = np.arange(0, num_time_steps * granularity, granularity).reshape(-1, 1)
    M = np.zeros_like(X, dtype=float)
    Delta = np.zeros_like(X, dtype=float)
    M[0, :] = 1  # first frame always observed

    for t in range(num_time_steps - 1):
        for d in range(num_variables):
            if X[t, d] != 'nan':
                M[t, d] = 1
                Delta[t + 1, d] = S[t + 1, 0] - S[t, 0]
            else:
                X[t, d] = 0
                Delta[t + 1, d] = S[t + 1, 0] - S[t, 0] + Delta[t, d]

    new_X = np.stack([X, M, Delta], axis=2)
    return new_X


# -----------------------------------------------------------------------------
# ForecastDataset definition
# -----------------------------------------------------------------------------

class ForecastDataset(torch_data.Dataset):
    """PyTorch Dataset for rolling‑window GRU‑D forecasting.

    Each sample contains:
    - `x`: input sequence of length `window_size+1`
    - `y`: prediction target sequence of length `horizon`
    - `para1`, `para2`: normalization parameters per node (mean/std)
    """

    def __init__(
        self,
        df,
        window_size: int,
        horizon: int,
        granularity: int,
        normalize_method: str | None = None,
        norm_statistic=None,
        interval: int = 1,
        norm_original=None,
    ):
        self.window_size = window_size + 1
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        self.norm_origin = norm_original
        self.granularity = granularity

        # Convert input DataFrame → numpy and mark missing as 'nan'
        df = pd.DataFrame(df).fillna('nan').values
        self.data = df
        self.df_length = len(df)

        # Determine slicing indices
        self.x_end_idx = self.get_x_end_idx()

        # Apply normalization if required
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

        # Apply GRU‑D augmentation: append mask & Δt channels
        self.data = grudData(self.data, self.granularity)

    def __getitem__(self, index: int):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size

        train_data = self.data[lo:hi]
        target_data = self.data[hi:hi + self.horizon]

        x = torch.from_numpy(train_data).float()
        y = torch.from_numpy(target_data).float()

        para1 = torch.from_numpy(np.asarray(self.norm_statistic['mean'])).float()
        para2 = torch.from_numpy(np.asarray(self.norm_statistic['std'])).float()

        return x, y, para1, para2

    def __len__(self) -> int:
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        """Return all valid upper indices for window extraction."""
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range(len(x_index_set) // self.interval)]
        return x_end_idx
