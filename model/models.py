import logging
from typing import Literal, Optional, Callable

import torch
from torch import nn, Tensor, autograd
from torch.nn import Module, functional as F

from . import graph_utils

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MLP(Module):
    """
    Implementation of a simple MLP model"""

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_out: int,
        num_layers: int,
        activation: nn.modules.activation = nn.Softplus(),
        activate_final=False,
        *args,
        **kwargs
    ):
        super().__init__()
        layers_list = [nn.Linear(dim_in, dim_hid), activation]
        for _ in range(num_layers - 1):
            layers_list += [nn.Linear(dim_hid, dim_hid), activation]
        layers_list += [nn.Linear(dim_hid, dim_out)]
        if activate_final:
            layers_list += [activation]
        self.model = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SingleMPNN(Module):
    """
    Implementation of a single MPNN model"""

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_out: int,
        edge_layers: int,
        node_layers: int,
        aggregation: Literal["sum", "mean"] = "sum",
        activation: nn.modules.activation = nn.Softplus(),
        activate_final: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        edge_layers_list = [nn.Linear(2 * dim_in, dim_hid), activation]
        for _ in range(edge_layers - 1):
            edge_layers_list += [nn.Linear(dim_hid, dim_hid), activation]
        self.edge_func = nn.Sequential(*edge_layers_list)

        node_layers_list = [nn.Linear(dim_in + dim_hid, dim_hid), activation]
        for _ in range(node_layers - 1):
            node_layers_list += [nn.Linear(dim_hid, dim_hid), activation]
        node_layers_list += [nn.Linear(dim_hid, dim_out)]
        if activate_final:
            node_layers_list += [activation]
        self.node_func = nn.Sequential(*node_layers_list)

        self.aggregation = (
            graph_utils.unsorted_segment_sum
            if aggregation == "sum"
            else graph_utils.unsorted_segment_mean
        )

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        batch_size, n = x.size(0), x.size(1)
        x = x.view(batch_size * n, -1)
        messages = self.edge_model(x, edge_idx)
        out = self.node_model(x, messages, edge_idx)
        out = out.view(batch_size, n, -1)
        return out

    def edge_model(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        """
        Phi function"""
        row, col = edge_idx[0], edge_idx[1]
        out = torch.cat([x[row], x[col]], dim=-1)
        out = self.edge_func(out)
        return out

    def node_model(self, x: Tensor, messages: Tensor, edge_idx: Tensor) -> Tensor:
        """
        Psi function"""
        row = edge_idx[0]
        out = self.aggregation(messages, row, x.size(0))
        out = torch.cat([x, out], dim=-1)
        out = self.node_func(out)
        return out


class PairwiseFunction(Module):
    """
    Implementation of a pairwise function, or summation of v_ij(x_i, x_j)"""

    def __init__(
        self,
        dim_in: int,
        dim_hid: int,
        dim_out: int,
        num_layers: int,
        activation: nn.modules.activation = nn.Softplus(),
        activate_final: bool = False,
        cofficient_func: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        super().__init__()
        layers_list = [nn.Linear(2 * dim_in, dim_hid), activation]
        for _ in range(num_layers - 1):
            layers_list += [nn.Linear(dim_hid, dim_hid), activation]
        layers_list += [nn.Linear(dim_hid, dim_out)]
        if activate_final:
            layers_list += [activation]
        self.model = nn.Sequential(*layers_list)
        self.cofficient_func = cofficient_func

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        row, col = edge_idx[0], edge_idx[1]
        out = torch.cat([x[row], x[col]], dim=-1)
        if self.cofficient_func is None:
            out = self.model(out)
        else:
            out = self.model(out) * self.cofficient_func(x, edge_idx)
        out = graph_utils.unsorted_segment_sum(out, row, x.size(0))
        return out


class C1Sigmoid(Module):
    """
    tilde{s}_a(x) in the paper"""

    def __init__(self, a: float):
        super().__init__()
        self.a = a
        self.a_half = a / 2.0
        self.square_coff = 2.0 / (a**2)

    def forward(self, x: Tensor) -> Tensor:
        return (
            torch.where(
                (x > 0) & (x < self.a_half),
                self.square_coff * x**2,
                torch.zeros_like(x),
            )
            + torch.where(
                (x >= self.a_half) & (x < self.a),
                torch.ones_like(x) - self.square_coff * (x - self.a) ** 2,
                torch.zeros_like(x),
            )
            + torch.where((x >= self.a), torch.ones_like(x), torch.zeros_like(x))
        )


class DistanceGraphCofficient(Module):
    """
    s_{ij}(x) in the paper for distance graph"""

    def __init__(self, delta: float, dim: int, alpha: float = 0.1):
        super().__init__()
        self.delta = delta
        self.dim = dim
        self.c1_sigmoid = C1Sigmoid(alpha * delta)

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        row, col = edge_idx[0], edge_idx[1]
        pos_relative = x[col, : self.dim] - x[row, : self.dim]
        dist = self.delta - torch.norm(pos_relative, dim=-1)
        return self.c1_sigmoid(dist).unsqueeze(-1)


class QuadraticFunction(Module):
    """
    Implementation of a quadratic function"""

    def __init__(self, epsilon: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x**2, dim=-1) * self.epsilon


class Vfunc(Module):
    """
    Implementation of a V function, summation of v_i(x_i),v_ij(x_i, x_j), and epsilon * x_i^2
    """

    def __init__(
        self, pair_funcs: list[Module], each_funcs: list[Module], *args, **kwargs
    ):
        super().__init__()
        self.pair_funcs = nn.ModuleList(pair_funcs)
        self.each_funcs = nn.ModuleList(each_funcs)

    def forward(self, x: Tensor, edge_idx: Tensor) -> Tensor:
        batch_size, n = x.size(0), x.size(1)
        x = x.view(batch_size * n, -1)
        out = torch.zeros(batch_size * n, device=x.device)
        for pair_func in self.pair_funcs:
            out += pair_func(x, edge_idx).squeeze(-1)
        for each_func in self.each_funcs:
            out += each_func(x).squeeze(-1)
        out = out.view(batch_size, n).sum(dim=-1)
        return out


class ProjectionModel(Module):
    """
    Implementation of a projection model"""

    def __init__(self, f_hat: SingleMPNN, v: Vfunc):
        super().__init__()
        self.f_hat = f_hat
        self.v = v

    def forward(
        self, x: Tensor, edge_idx: Tensor, check=False, output_v=False
    ) -> Tensor:
        f_hat_x = self.f_hat(x, edge_idx)
        v_x = self.v(x, edge_idx)
        grad_v = autograd.grad(v_x.sum(dim=-1), x, create_graph=True)[0]
        batch_size, n = x.shape[:2]
        f_hat_x = f_hat_x.view(batch_size * n, -1)
        grad_v = grad_v.view(batch_size * n, -1)
        f_x = f_hat_x - grad_v * graph_utils.non_zero_div(
            F.relu((f_hat_x * grad_v).sum(dim=-1)), (grad_v**2).sum(dim=-1)
        ).unsqueeze(-1)
        f_x = f_x.view(batch_size, n, -1)
        if check:
            g_dot = (f_x * grad_v).sum(dim=-1)
            if torch.any(g_dot > 0):
                logger.warning(r"$dot{V}>0$", g_dot)
        if output_v:
            return f_x, v_x
        return f_x


class HamiltonianModel(Module):
    """
    Implementation of a Hamiltonian model"""

    def __init__(self, j_mat: SingleMPNN, r_mat: SingleMPNN, v: Vfunc, dim: int):
        super().__init__()
        self.j_mat = j_mat
        self.r_mat = r_mat
        self.v = v
        self.dim = dim

    def forward(
        self, x: Tensor, edge_idx: Tensor, check=False, output_v=False
    ) -> Tensor:
        j_x = self.j_mat(x, edge_idx)
        r_x = self.r_mat(x, edge_idx)
        v_x = self.v(x, edge_idx)
        grad_v = autograd.grad(v_x.sum(dim=-1), x, create_graph=True)[0]
        batch_size, n = x.shape[:2]
        j_x = j_x.view(batch_size * n, self.dim, self.dim)
        j_skew = j_x.transpose(1, 2) - j_x
        r_x = r_x.view(batch_size * n, self.dim, self.dim)
        r_pos = r_x.transpose(1, 2) @ r_x
        grad_v = grad_v.view(batch_size * n, -1)
        f_x = ((j_skew - r_pos) @ grad_v.unsqueeze(-1)).view(batch_size, n, -1)
        if check:
            g_dot = (f_x * grad_v).sum(dim=-1)
            if torch.any(g_dot > 0):
                logger.warning(r"$dot{V}>0$", g_dot)
        if output_v:
            return f_x, v_x
        return f_x


# Hereafter, the functions are not used for experiments in the paper


class RectangularLogBarrierFunction(Module):
    """
    Implementation of a rectangular log barrier function"""

    def __init__(
        self, bounds: list[list[float]], epsilon: float = 1e-3, *args, **kwargs
    ):
        super().__init__()
        self.bounds = bounds
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        log = 0
        for i in range(len(self.bounds)):
            log += torch.log(x[:, i] - self.bounds[i][0]) + torch.log(
                self.bounds[i][1] - x[:, i]
            )
        return -log * self.epsilon


class RectangularInverseBarrierFunction(Module):
    """
    Implementation of a rectangular inverse barrier function"""

    def __init__(
        self, bounds: list[list[float]], epsilon: float = 1e-3, *args, **kwargs
    ):
        super().__init__()
        self.bounds = bounds
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        inv = 0
        for i in range(len(self.bounds)):
            inv += 1 / (x[:, i] - self.bounds[i][0]) + 1 / (self.bounds[i][1] - x[:, i])
        return -inv * self.epsilon
