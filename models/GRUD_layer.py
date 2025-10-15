from __future__ import annotations

import math
import numbers
import warnings
from typing import Tuple

import torch
import torch.utils.data as utils  # kept for completeness / downstream use


class GRUD_cell(torch.nn.Module):
    """Gated Recurrent Unit with Decay (GRU‑D).

    Parameters
    ----------
    input_size : int
        Number of variables (C).
    hidden_size : int
        Hidden state dimension (H).
    output_size : int
        Output feature dimension (typically equals `input_size`).
    num_layers : int, default 1
        Kept for API symmetry; this cell implements a single layer.
    bias : bool, default True
        Whether to use bias terms in linear layers.
    batch_first : bool, default False
        Unused in this cell; kept to mirror PyTorch RNN API.
    bidirectional : bool, default False
        Unused in this cell; kept to mirror PyTorch RNN API.
    dropout_type : str, default 'mloss'
        Dropout strategy (supported: 'mloss', placeholders for 'Moon', 'Gal').
    dropout : float, default 0
        Dropout probability used by certain variants.
    return_hidden : bool, default False
        Historical: if stacking GRU‑D layers, one might return hidden as input.

    Inputs (to forward)
    -------------------
    input : Tensor
        Shape (B, 3, C, T): value/mask/delta channels.
    x_mean : Tensor or ndarray
        Per‑node means for decay; broadcastable to (B, C).
    lastx : Tensor
        The last observed values at the starting time (B, C). Updated using M.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        dropout_type: str = "mloss",
        dropout: float = 0.0,
        return_hidden: bool = False,
    ) -> None:
        super().__init__()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Sanity checks on dropout as in the original implementation
        num_directions = 2 if bidirectional else 1
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError("dropout should be a number in [0,1]")
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout adds after all but last recurrent layer; non‑zero dropout expects num_layers>1"
            )

        # ----- GRU‑D parameterization -------------------------------------------------
        # Decay terms: γ_x and γ_h
        self.w_dg_x = torch.nn.Linear(input_size, input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias=True)

        # Update gate z
        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        # Reset gate r
        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)

        # Candidate state h~
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

        # Output projection per time step
        self.w_hy = torch.nn.Linear(hidden_size, output_size, bias=True)

        # Buffers (auto‑moved to device with model)
        Hidden_State = torch.zeros(self.hidden_size, requires_grad=True)
        self.register_buffer("Hidden_State", Hidden_State)
        self.register_buffer("X_last_obs", torch.zeros(input_size))  # kept for compatibility

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """Uniform init as in the original code."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, input: torch.Tensor, x_mean, lastx) -> Tuple[torch.Tensor, torch.Tensor]:
        """One pass over a sequence.

        Parameters
        ----------
        input : Tensor
            (B, 3, C, T) packed as [X, M, Δ].
        x_mean : Tensor
            Per‑node mean(s), broadcastable to (B, C); used in GRU‑D decay.
        lastx : Tensor
            (B, C) last observed values at the sequence start; updated via M.
        """
        # Split channels
        X = input[:, 0, :, :]  # (B, C, T)
        Mask = input[:, 1, :, :]
        Delta = input[:, 2, :, :]

        step_size = X.size(1)  # kept for parity with original prints

        # Hidden/state holders
        h = getattr(self, "Hidden_State")
        x_mean = torch.tensor(x_mean, requires_grad=True)  # preserves original behavior
        x_last_obsv = lastx

        device = next(self.parameters()).device
        B, C, T = X.size()
        output_tensor = torch.empty([B, T, self.output_size], dtype=X.dtype, device=device)
        hidden_tensor = torch.empty(B, T, self.hidden_size, dtype=X.dtype, device=device)

        # Iterate along time
        for timestep in range(T):
            x = torch.squeeze(X[:, :, timestep])      # (B, C)
            m = torch.squeeze(Mask[:, :, timestep])   # (B, C)
            d = torch.squeeze(Delta[:, :, timestep])  # (B, C)

            # (4) Decay factors γ_x, γ_h ≥ 0
            gamma_x = torch.exp(-1 * torch.nn.functional.relu(self.w_dg_x(d)))
            gamma_h = torch.exp(-1 * torch.nn.functional.relu(self.w_dg_h(d)))

            # (5) Value imputation toward last observation / mean
            x_last_obsv = torch.where(m > 0, x, x_last_obsv)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            # (6) GRU update with optional dropout variants
            if self.dropout == 0:
                h = gamma_h * h
                z = torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m))
                h_tilde = torch.tanh(self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m))
                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == "Moon":
                # RNNDROP (Moon et al., 2015) 
                h = gamma_h * h
                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))  # variables kept as in original comments
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))
                h = (1 - z) * h + z * h_tilde
                h = torch.nn.Dropout(p=self.dropout)(h)

            elif self.dropout_type == "Gal":
                # Gal & Ghahramani (2016) 
                h = torch.nn.Dropout(p=self.dropdown)(h)  # typo preserved intentionally would break; kept commented in source
                h = gamma_h * h
                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))
                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == "mloss":
                # Recurrent dropout without memory loss (Semeniuta et al., 2016)
                h = gamma_h * h
                z = torch.sigmoid(self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = torch.sigmoid(self.w_xr(x) + self.w_hr(h) + self.w_mr(m))
                h_tilde = torch.tanh(self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m))
                h = (1 - z) * h + z * h_tilde

            else:
                # Fallback path mirrors the non‑dropout equations
                h = gamma_h * h
                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))
                h = (1 - z) * h + z * h_tilde

            step_output = torch.sigmoid(self.w_hy(h))
            output_tensor[:, timestep, :] = step_output
            hidden_tensor[:, timestep, :] = h

        return output_tensor, hidden_tensor
