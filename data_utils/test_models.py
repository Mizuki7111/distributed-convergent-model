import numpy as np
from numpy import ndarray
import torch
from torch.nn import Module

from typing import Optional, Callable


def long_term_prediction(
    model: Module,
    data: np.ndarray,
    steps: int,
    dt: float = 0.01,
    graph: Optional[ndarray] = None,
    graph_generator: Optional[Callable] = None,
    require_grad: bool = True,
) -> ndarray:
    """
    Test the trained model by long-term prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.zeros((data.shape[0], steps, *data.shape[1:]))
    if graph is None:
        if graph_generator is None:
            raise ValueError("graph or graph_generator must be provided.")
    data = torch.from_numpy(data).to(torch.float32)
    model.to(device)
    model.eval()
    result[:, 0] = data
    for stp in range(1, steps):
        if graph_generator is not None:
            graph = graph_generator(data.to("cpu"))
        n = data.shape[1]
        graph_ = [g + bi * n for bi, g in enumerate(graph)]
        graph_ = torch.cat(graph_, dim=1)
        data = data.to(device).requires_grad_(True) if require_grad else data.to(device)
        graph_ = graph_.to(device)
        data = data + model(data, graph_) * dt
        result[:, stp] = data
    return result.cpu().detach().numpy()


def test_v_pair(v_pair: Module, range: list, steps: int):
    """
    Map the scalar function v_pair with d=1 to a 2D grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(range[0], range[1], steps)
    y = torch.linspace(range[0], range[1], steps)
    x, y = torch.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    data = torch.cat([x, y]).unsqueeze(-1)
    row = torch.cat([torch.arange(steps**2), torch.arange(steps**2) + steps**2])
    col = torch.cat([torch.arange(steps**2) + steps**2, torch.arange(steps**2)])
    edge_idx = torch.stack([row, col], dim=0)
    v_pair.to(device)
    v_pair.eval()
    data = data.to(device)
    edge_idx = edge_idx.to(device)
    v = v_pair(data, edge_idx)[: steps**2] + v_pair(data, edge_idx)[steps**2 :]
    x = x.view(steps, steps).cpu().detach().numpy()
    y = y.view(steps, steps).cpu().detach().numpy()
    v = v.view(steps, steps).cpu().detach().numpy()
    return {"x": x, "y": y, "v": v}


def test_v_self(v_self, range: list, steps: int):
    """
    Map the scalar function v_self with d=1 to a 1D grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(range[0], range[1], steps).unsqueeze(-1)
    v_self.to(device)
    v_self.eval()
    x = x.to(device)
    v = v_self(x)
    x = x.cpu().detach().numpy()
    v = v.cpu().detach().numpy()
    return {"x": x, "v": v}
