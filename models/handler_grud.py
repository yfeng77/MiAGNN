from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F
from numpy import savetxt
from data_loader.forecast_dataloader_grud import (
    ForecastDataset,
    de_normalized,
    normalized,
    de_normalized_batch,
)
from models.MiAGNN import Model
from utils.math_utils import evaluate


use_cuda = torch.cuda.is_available()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def split_data(args, data: np.ndarray):
    """Split a full series into train/valid/test by ratios on `args`.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Must define `train_length`, `valid_length`, `test_length`.
    data : np.ndarray, shape (T, N)
        Full time series matrix.

    Returns
    -------
    (train, valid, test) : tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
    valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
    test_ratio = 1 - train_ratio - valid_ratio

    train_data = data[: int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)) : int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)) :]
    return train_data, valid_data, test_data


def save_model(model: nn.Module, model_dir: str | None, epoch: int | None = None) -> None:
    """Save a full PyTorch model object into `model_dir`.

    The file is named `{epoch}_stemgnn.pt` when `epoch` is provided,
    otherwise `'_stemgnn.pt'`.
    """
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch_str = str(epoch) if epoch is not None else ""
    file_name = os.path.join(model_dir, f"{epoch_str}_stemgnn.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)


def load_model(model_dir: str | None, epoch: int | None = None) -> nn.Module | None:
    """Load a full PyTorch model object from `model_dir`.

    Returns `None` if the file does not exist. Mirrors `save_model` naming.
    """
    if not model_dir:
        return None
    epoch_str = str(epoch) if epoch is not None else ""
    file_name = os.path.join(model_dir, f"{epoch_str}_stemgnn.pt")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return None
    with open(file_name, "rb") as f:
        model = torch.load(f)
    return model


# -----------------------------------------------------------------------------
# Inference / Validation
# -----------------------------------------------------------------------------

def inference(
    model: nn.Module,
    dataloader: torch_data.DataLoader,
    device: torch.device,
    node_cnt: int,
    window_size: int,
    horizon: int,
    args,
):
    """Autoregressive, batched multi‑step inference.

    Workflow per batch:
    1) Extract `lastx` (the very first frame) from channel 0.
    2) Optionally simulate missing nodes (uniform at random) according to
       `args.missing_rate` by zeroing features and filling the \u0394t channel.
    3) Run the model until we accumulate `horizon` steps.
    4) Collect both normalized predictions and, for MAPSS, min‑max de‑scaled
       versions for reporting.

    Shapes
    ------
    - inputs:   (B, T_w+1, N, C)  -> we drop the first frame before AR loop
    - target:   (B, H,    N, C)
    - forecast: (B, H,    N)
    """

    forecast_set = []
    target_set = []
    forecast_rescale_set = []
    target_rescale_set = []

    model.eval()
    with torch.no_grad():
        for i, (inputs, target, scale1, scale2) in enumerate(dataloader):
            # `lastx` keeps the very first observation (channel 0 only).
            lastx = inputs[:, 0, :, 0].to(args.device)
            # Drop the first frame; remaining window is length `window_size`.
            inputs = inputs[:, 1:, :, :]  # (B, T_w, N, C)

            # --- Simulate missing nodes for this batch ---------------------
            index = random.sample(range(args.node_count), int(args.missing_rate * args.node_count))
            inputs[:, :, index, :] = 0.0  # zero features for masked nodes

            # Fill GRU‑D style delta‑time (minutes since last obs) in channel 2.
            delta = np.arange(args.granularity, (args.window_size + 1) * args.granularity, args.granularity)
            # Shape we want for broadcasting: (B, T_w, |index|)
            delta_extend = np.tile(delta, (inputs.shape[0], len(index), 1)).transpose(0, 2, 1)
            inputs[:, :, index, 2] = torch.from_numpy(delta_extend).float()

            # Per‑node mean over the window (channel 0); fed as a conditioning
            # statistic to the model.
            x_mean = inputs[:, :, :, 0].mean(dim=1).mean(dim=0)  # (N,)
            x_mean = torch.as_tensor(x_mean, device=args.device)

            inputs = inputs.to(device)
            target = target.to(device)
            target = target[:, :, :, 0]  # regression target is channel 0

            # Autoregressive roll‑out to reach `horizon` steps.
            step = 0
            forecast_steps = np.zeros((inputs.size(0), horizon, node_cnt), dtype=np.float64)
            while step < horizon:
                forecast_result, cl_result = model(inputs, x_mean, lastx, index)
                len_model_output = forecast_result.size(1)
                if len_model_output == 0:
                    raise RuntimeError("Empty inference result from model")

                # Slide the input window and append the new predictions on ch 0
                inputs[:, : window_size - len_model_output, :, 0] = inputs[:, len_model_output:window_size, :, 0].clone()
                inputs[:, window_size - len_model_output :, :, 0] = forecast_result.clone()

                take = min(horizon - step, len_model_output)
                forecast_steps[:, step : step + take, :] = (
                    forecast_result[:, :take, :].detach().cpu().numpy()
                )
                step += take

            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())

            # Optionally return de‑normalized values for MAPSS
            if args.dataset == "MAPSS":
                forecast_steps_rescale = de_normalized_batch(
                    forecast_steps,
                    args.norm_method,
                    scale1.detach().cpu().numpy(),
                    scale2.detach().cpu().numpy(),
                )
                forecast_rescale_set.append(forecast_steps_rescale)
                target_steps_rescale = de_normalized_batch(
                    target.detach().cpu().numpy(),
                    args.norm_method,
                    scale1.detach().cpu().numpy(),
                    scale2.detach().cpu().numpy(),
                )
                target_rescale_set.append(target_steps_rescale)

    if args.dataset == "MAPSS":
        return (
            np.concatenate(forecast_set, axis=0),
            np.concatenate(target_set, axis=0),
            np.concatenate(forecast_rescale_set, axis=0),
            np.concatenate(target_rescale_set, axis=0),
        )
    else:
        return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(
    args,
    model: nn.Module,
    dataloader: torch_data.DataLoader,
    device: torch.device,
    normalize_method: str | None,
    statistic: dict | None,
    node_cnt: int,
    window_size: int,
    horizon: int,
    result_file: str | None = None,
):
    """Run inference and compute error metrics (MAE/MAPE/RMSE).

    Returns
    -------
    forecast_norm, target_norm, metrics_dict
        The first two are *normalized* arrays (for logging). For MAPSS we also
        compute and evaluate on de‑normalized values.
    """
    start = datetime.now()
    if args.dataset == "MAPSS":
        forecast_norm, target_norm, forecast, target = inference(
            model, dataloader, device, node_cnt, window_size, horizon, args
        )
        score_norm = evaluate(target_norm, forecast_norm)
        score = evaluate(target, forecast)
    else:
        forecast_norm, target_norm = inference(
            model, dataloader, device, node_cnt, window_size, horizon, args
        )
        if normalize_method and statistic:
            forecast = de_normalized(forecast_norm, normalize_method, statistic)
            target = de_normalized(target_norm, normalize_method, statistic)
        else:
            forecast, target = forecast_norm, target_norm
        score_norm = evaluate(target_norm, forecast_norm)
        score = evaluate(target, forecast)
    end = datetime.now()

    return (
        forecast_norm,
        target_norm,
        dict(
            mae=score[1],
            mape=score[0],
            rmse=score[2],
            mae_norm=score_norm[1],
            mape_norm=score_norm[0],
            rmse_norm=score_norm[2],
        ),
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(train_data, valid_data, args, result_file: str):
    """Main training loop.

    Parameters
    ----------
    train_data, valid_data : np.ndarray or DataLoader
        When `args.dataset != 'MAPSS'`, these are raw arrays `(T, N)` and will
        be wrapped by `ForecastDataset`. Otherwise, pass prebuilt loaders.
    args : namespace
        Expects fields such as `window_size`, `horizon`, `batch_size`,
        `optimizer`, `lr`, `decay_rate`, `validate_freq`, `exponential_decay_step`,
        `node_count`, `missing_rate`, `granularity`, `norm_method`, etc.
    result_file : str
        Directory to save checkpoints and normalization stats.
    """
    node_cnt = args.node_count

    if len(train_data) == 0:
        raise RuntimeError("Cannot organize enough training data")
    if len(valid_data) == 0:
        raise RuntimeError("Cannot organize enough validation data")

    # Build datasets/loaders (or forward the given ones for MAPSS)
    if args.dataset != "MAPSS":
        if args.norm_method == "z_score":
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        elif args.norm_method == "min_max":
            train_min = np.min(train_data, axis=0)
            train_max = np.max(train_data, axis=0)
            normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
        else:
            normalize_statistic = None

        if normalize_statistic is not None:
            os.makedirs(result_file, exist_ok=True)
            with open(os.path.join(result_file, "norm_stat.json"), "w") as f:
                json.dump(normalize_statistic, f)

        train_set = ForecastDataset(
            train_data,
            window_size=args.window_size,
            horizon=args.horizon,
            granularity=args.granularity,
            normalize_method=args.norm_method,
            norm_statistic=normalize_statistic,
        )
        valid_set = ForecastDataset(
            valid_data,
            window_size=args.window_size,
            horizon=args.horizon,
            granularity=args.granularity,
            normalize_method=args.norm_method,
            norm_statistic=normalize_statistic,
        )
        train_loader = torch_data.DataLoader(
            train_set, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=4
        )
        valid_loader = torch_data.DataLoader(
            valid_set, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=4
        )
    else:
        train_loader = train_data
        valid_loader = valid_data
        normalize_statistic = None

    # Model
    hidden_size = 44  
    model = Model(
        node_cnt,
        2,
        args.window_size,
        args.multi_layer,
        horizon=args.horizon,
        hidden_size=hidden_size,
        dropout_type="mloss",
    ).to(args.device)

    # Optimizer / scheduler
    if args.optimizer == "RMSProp":
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    forecast_loss = nn.MSELoss(reduction="mean").to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}

    loss_values = []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0.0
        cnt = 0

        for i, (inputs, target, _, _) in enumerate(train_loader):
            # First frame of channel 0 is used as `lastx`
            lastx = inputs[:, 0, :, 0].to(args.device)
            # Discard the first frame; keep a fixed window for AR updates
            inputs = inputs[:, 1:, :, :]

            # --- Build train‑time masks and GRU‑D delta channel ------------
            index = random.sample(range(args.node_count), int(args.missing_rate * args.node_count))
            y = list(range(args.node_count))
            index_nm = torch.IntTensor(list(set(y) - set(index))).to(args.device)  # not used but kept for reference

            inputs[:, :, index, :] = 0
            delta = np.arange(args.granularity, (args.window_size + 1) * args.granularity, args.granularity)
            delta_extend = np.tile(delta, (inputs.shape[0], len(index), 1)).transpose(0, 2, 1)
            inputs[:, :, index, 2] = torch.from_numpy(delta_extend).float()

            # Per‑node mean over the window (channel 0)
            x_mean = inputs[:, :, :, 0].mean(dim=1).mean(dim=0)
            x_mean = torch.as_tensor(x_mean, device=args.device)

            inputs = inputs.to(args.device)

            # ctarget is the classification target for the (optional) mask head
            ctarget = target[:, :, :, 1].to(args.device)
            ctarget[:, :, index] = 0  # masked nodes labeled as zero

            # Regression target is channel 0
            target = target.to(args.device)  # (B, H, N, C)
            target = target[:, :, :, 0]

            model.zero_grad()
            forecast, logits = model(inputs, x_mean, lastx, index)

            # Two‑part loss: regression + (weighted) classification
            loss = (1 - args.missing_rate) * forecast_loss(forecast, target) + args.missing_rate * F.cross_entropy(logits, ctarget)

            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        print(
            "| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}".format(
                epoch, (time.time() - epoch_start_time), loss_total / max(cnt, 1)
            )
        )
        loss_values.append(loss_total / max(cnt, 1))

        # Save epoch checkpoint
        save_model(model, result_file, epoch)

        # LR decay
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()

        # Periodic validation + best model tracking
        if (epoch + 1) % args.validate_freq == 0:
            _, _, performance_metrics = validate(
                args,
                model,
                valid_loader,
                args.device,
                args.norm_method,
                normalize_statistic,
                node_cnt,
                args.window_size,
                args.horizon,
                result_file=result_file,
            )
            if best_validate_mae > performance_metrics["mae"]:
                best_validate_mae = performance_metrics["mae"]
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                is_best_for_now = False
                validate_score_non_decrease_count += 1

            if is_best_for_now:
                save_model(model, result_file)  # save as the unnamed "best"

        # Early stopping
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break

    # Persist the training loss curve for later plotting
    savetxt(f"MA{args.missing_rate}{args.dataset}loss.txt", loss_values, delimiter=",")
    return performance_metrics, normalize_statistic


# -----------------------------------------------------------------------------
# Plotting / Testing
# -----------------------------------------------------------------------------

def plot_multivariate_forecast(data, mr, input, y_true, y_pred, node_idx: int):
    """Plot ground truth vs prediction for one node from the last batch.

    Parameters
    ----------
    data : str
        Dataset name (used in the figure filename).
    mr : float
        Missing rate (used in the figure filename).
    input : np.ndarray or torch.Tensor
        Input window; expects shape `(B, T_w, N, C)`.
    y_true : np.ndarray or torch.Tensor
        Ground truth of shape `(B, H, N, C)`.
    y_pred : np.ndarray
        Model predictions of shape `(B, H, N)` (normalized/de‑normalized).
    node_idx : int
        Index of the node to visualize.
    """
    plt.figure(figsize=(12, 6))

    plt.plot(
        np.concatenate((input[-1, :, node_idx, 0], y_true[-1, :, node_idx, 0])),
        label=f"Ground Truth Batch {0 + 1}",
        linestyle="--",
    )
    plt.plot(
        np.concatenate((input[-1, :, node_idx, 0], y_pred[-1, :, node_idx])),
        label=f"Prediction Batch {0 + 1}",
    )

    plt.title(f"Time Series Forecasting for Node {node_idx}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{data}{mr}Node{node_idx}.png")
    plt.show()


def test(test_data, args, result_train_file: str, result_test_file: str):
    """Evaluate on the test split and print metrics.

    Returns
    -------
    mae, rmse, mae_norm, rmse_norm : float
        Raw and normalized error metrics for convenient logging.
    """
    if (args.norm_method in {"z_score", "min_max"}) and (args.dataset != "MAPSS"):
        with open(os.path.join(result_train_file, "norm_stat.json"), "r") as f:
            normalize_statistic = json.load(f)
    else:
        normalize_statistic = None

    model = load_model(result_train_file)

    node_cnt = args.node_count

    if args.dataset != "MAPSS":
        test_set = ForecastDataset(
            test_data,
            window_size=args.window_size,
            horizon=args.horizon,
            granularity=args.granularity,
            normalize_method=args.norm_method,
            norm_statistic=normalize_statistic,
        )
        test_loader = torch_data.DataLoader(
            test_set, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=0
        )
    elif args.dataset == "MAPSS":
        test_loader = test_data
    else:
        raise ValueError("No available dataset.")

    _, _, performance_metrics = validate(
        args,
        model,
        test_loader,
        args.device,
        args.norm_method,
        normalize_statistic,
        node_cnt,
        args.window_size,
        args.horizon,
        result_file=result_test_file,
    )
    mae, mape, rmse = (
        performance_metrics["mae"],
        performance_metrics["mape"],
        performance_metrics["rmse"],
    )
    mae_norm, mape_norm, rmse_norm = (
        performance_metrics["mae_norm"],
        performance_metrics["mape_norm"],
        performance_metrics["rmse_norm"],
    )
    print(
        "Performance on test set: MAPE: {:5.4f} | MAE: {:5.4f} | RMSE: {:5.4f}".format(mape, mae, rmse)
    )
    print(
        "Performance on test set: MAPE_norm: {:5.4f} | MAE_norm: {:5.4f} | RMSE_norm: {:5.4f}".format(
            mape_norm, mae_norm, rmse_norm
        )
    )

    return mae, rmse, mae_norm, rmse_norm


# -----------------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------------

def load_solar(args):
    """Load Solar dataset from `dataset/{args.dataset}.txt`.

    Sets `args.node_count` to the number of sensors (columns).
    """
    data = np.loadtxt(os.path.join("dataset", args.dataset + ".txt"), delimiter=",")
    args.node_count = data.shape[1]
    return data


def load_elec(args):
    """Load Electricity dataset from `dataset/{args.dataset}.txt`."""
    data = np.loadtxt(os.path.join("dataset", args.dataset + ".txt"), delimiter=",")
    args.node_count = data.shape[1]
    return data


def load_mapss_sub(args, name: str):
    """Helper to construct a DataLoader for a MAPSS (RUL) split.

    Steps
    -----
    1) Read `dataset/{name}` (space‑delimited numeric txt).
    2) Convert to DataFrame and select `sensor_1..sensor_21`.
    3) For each unit (engine), compute a per‑unit min‑max normalization,
       then optionally a z‑score normalization for the learning pipeline.
    4) Wrap each unit as `ForecastDataset` and concatenate.
    """
    data = np.loadtxt(os.path.join("dataset", name))
    columns = [
        "unit",
        "time",
        "setting_1",
        "setting_2",
        "setting_3",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_10",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_16",
        "sensor_17",
        "sensor_18",
        "sensor_19",
        "sensor_20",
        "sensor_21",
    ]
    df = pd.DataFrame(data, columns=columns)

    sensor = df.loc[:, "sensor_1":"sensor_21"].columns

    grouped = df.groupby(df.unit)
    print("# series:", len(grouped))

    all_data = []
    for i in range(len(grouped) - 1):
        series_id = float(i + 1)
        df_lot = grouped.get_group(series_id)
        df_data = df_lot[sensor]
        data = df_data.values

        # Unit‑level min‑max normalization (original scale -> [0,1])
        data_min_ori = np.min(data, axis=0)
        data_max_ori = np.max(data, axis=0)
        normalize_statistic_ori = {"min": data_min_ori.tolist(), "max": data_max_ori.tolist()}
        data_scaled, _ = normalized(data, normalize_method="min_max", norm_statistic=normalize_statistic_ori)

        # (Optional) additional z‑score normalization for learning
        if args.norm_method == "z_score":
            data_mean = np.mean(data_scaled, axis=0)
            data_std = np.std(data_scaled, axis=0)
            normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        else:
            normalize_statistic = None

        data_set = ForecastDataset(
            data_scaled,
            window_size=args.window_size,
            horizon=args.horizon,
            normalize_method=args.norm_method,
            norm_statistic=normalize_statistic,
            norm_original=normalize_statistic_ori,
        )
        all_data.append(data_set)

    data_set_all = torch.utils.data.ConcatDataset(all_data)
    loader = torch.utils.data.DataLoader(
        data_set_all, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=0
    )

    return loader


def load_MAPSS(args):
    """Return DataLoaders for MAPSS train/valid/test splits."""
    args.node_count = 21
    train_loader = load_mapss_sub(args, "train.txt")
    valid_loader = load_mapss_sub(args, "test.txt")
    test_loader = load_mapss_sub(args, "final_test.txt")
    return train_loader, valid_loader, test_loader


def load_PeMS07(args):
    """Load PeMS07 traffic dataset from CSV and set `args.node_count`."""
    data_file = os.path.join("dataset", args.dataset + ".csv")
    data = pd.read_csv(data_file).values  # shape: (T, N)
    print("pd.read_csv(data_file).values", data.shape)  # e.g., (12671, 228)
    args.node_count = data.shape[1]
    return data
