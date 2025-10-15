import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GRUD_layer import GRUD_cell


# -----------------------------------------------------------------------------
# Multi-task loss weighting
# -----------------------------------------------------------------------------
class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss from Kendall et al. (2018)."""

    def __init__(self, num: int = 2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num, requires_grad=True))

    def forward(self, *x):
        loss_sum = 0.0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


# -----------------------------------------------------------------------------
# Basic Graph Convolution / GCN Head
# -----------------------------------------------------------------------------
class GraphConvolution(nn.Module):
    """Graph convolution based on multi-order Laplacians."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(4, 1, input_size, output_size))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        L = L.unsqueeze(1)
        x = torch.matmul(L, x)
        x = torch.matmul(x, self.weight)
        return torch.sum(x, dim=0)


class GCN(nn.Module):
    """Two-layer GCN with ReLU and sigmoid activations."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, 3)

    def forward(self, x, L):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        L = L.to(device)
        x = F.relu(self.gc1(x, L))
        return torch.sigmoid(self.gc2(x, L))


# -----------------------------------------------------------------------------
# GLU and StockBlock for Spectral Modeling
# -----------------------------------------------------------------------------
class GLU(nn.Module):
    """Gated Linear Unit (GLU) used within spectral graph layers."""

    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_left(x) * torch.sigmoid(self.linear_right(x))


class StockBlockLayer(nn.Module):
    """Spectral convolutional block for graph signals (frequency-domain GCN)."""

    def __init__(self, time_step: int, unit: int, multi_layer: int, stack_cnt: int = 0):
        super().__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer

        self.weight = nn.Parameter(
            torch.Tensor(1, 4, 1, self.time_step * self.multi, self.multi * self.time_step)
        )
        nn.init.xavier_normal_(self.weight)

        # Forecast / backcast projections
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)

        self.relu = nn.ReLU()

        # GLU stack for frequency transformation
        self.output_channel = 4 * self.multi
        self.GLUs = nn.ModuleList()
        for i in range(3):
            in_dim = self.time_step * (4 if i == 0 else self.output_channel)
            out_dim = self.time_step * self.output_channel
            self.GLUs.append(GLU(in_dim, out_dim))
            self.GLUs.append(GLU(in_dim, out_dim))

    def spe_seq_cell(self, input: torch.Tensor) -> torch.Tensor:
        """Spectral-domain convolution: FFT + GLU + IFFT."""
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)

        # FFT along temporal dimension
        ffted = torch.view_as_real(torch.fft.fft(input, dim=1))
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        # Apply three pairs of GLU transformations
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[i * 2 + 1](img)

        # Recombine and inverse FFT
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        return torch.fft.irfft(torch.view_as_complex(time_step_as_inner), n=time_step_as_inner.shape[1], dim=1)

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)

        gfted = torch.matmul(mul_L, x)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)

        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)

        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None

        return forecast, backcast_source


# -----------------------------------------------------------------------------
# Main MiAGNN Model
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """Missing-aware Adaptive Graph Neural Network (MiAGNN)."""

    def __init__(
        self,
        units: int,
        stack_cnt: int,
        time_step: int,
        multi_layer: int,
        hidden_size: int,
        horizon: int,
        x_mean: float = 0.0,
        dropout_type: str = 'mloss',
        dropout_rate: float = 0.5,
        leaky_rate: float = 0.2,
        device: str = 'cpu',
    ):
        super().__init__()

        self.unit = units
        self.stack_cnt = stack_cnt
        self.time_step = time_step
        self.horizon = horizon
        self.alpha = leaky_rate

        # Graph attention weights
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        # Temporal encoder (GRU-D + GRU)
        self.gru_d = GRUD_cell(input_size=self.unit, hidden_size=hidden_size, output_size=self.unit, dropout=0, dropout_type=dropout_type)
        self.GRU = nn.GRU(self.time_step, self.unit)

        # Frequency-domain graph convolutional stack
        self.stock_block = nn.ModuleList([
            StockBlockLayer(self.time_step, self.unit, multi_layer, stack_cnt=i) for i in range(self.stack_cnt)
        ])

        # Forecast projection
        self.fc = nn.Sequential(
            nn.Linear(self.time_step, self.time_step),
            nn.LeakyReLU(),
            nn.Linear(self.time_step, self.horizon),
        )

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.GCN = GCN(3, 16)
        self.to(device)

    # ---------------------------------------------------------------------
    # Graph operations
    # ---------------------------------------------------------------------
    def get_laplacian(self, graph, normalize=True):
        """Compute graph Laplacian (normalized or unnormalized)."""
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
            return torch.eye(graph.size(0), device=graph.device) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            return D - graph

    def cheb_polynomial(self, laplacian: torch.Tensor) -> torch.Tensor:
        """Compute Chebyshev polynomials up to order 3."""
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first = torch.zeros([1, N, N], device=laplacian.device)
        second = laplacian
        third = 2 * torch.matmul(laplacian, second) - first
        fourth = 2 * torch.matmul(laplacian, third) - second
        return torch.cat([first, second, third, fourth], dim=0)

    # ---------------------------------------------------------------------
    # Graph Attention
    # ---------------------------------------------------------------------
    def self_graph_attention(self, input: torch.Tensor, index):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2).view(bat, N, -1)
        attention = F.softmax(self.leakyrelu(data), dim=2)
        return self.dropout(attention)

    def latent_correlation_layer(self, x, x_mean, lastx, index):
        """Compute latent adjacency via GRU-D + GRU + self-attention."""
        input, _ = self.gru_d(x.permute(0, 3, 2, 1).contiguous(), x_mean, lastx)
        input, _ = self.GRU(input.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()

        attention = torch.mean(self.self_graph_attention(input, index), dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diag_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diag_hat, torch.matmul(degree_l - attention, diag_hat))
        return self.cheb_polynomial(laplacian), attention

    # ---------------------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------------------
    def forward(self, x, x_mean, lastx, index):
        mul_L, attention = self.latent_correlation_layer(x, x_mean, lastx, index)
        X = x[:, :, :, 0].unsqueeze(1).permute(0, 1, 3, 2).contiguous()

        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)

        forecast = sum(result)
        forecast = self.fc(forecast)
        output = self.GCN(forecast, mul_L)

        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), output.unsqueeze(1).squeeze(-1)
        else:
            return forecast.permute(0, 2, 1).contiguous(), output.permute(0, 2, 1).contiguous()
