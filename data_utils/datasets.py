import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable


class DistanceGraphTorch:
    """
    Generate graph based on distance between agents"""

    def __init__(self, distance: float, dim: int):
        self.distance = distance
        self.dim = dim

    def __call__(self, data: Tensor, *args, **kwds) -> list[Tensor]:
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        pos = data[:, :, : self.dim]
        len_data = pos.size(0)
        graph_list = []
        for i in range(len_data):
            pos_relative = pos[i].unsqueeze(0) - pos[i].unsqueeze(1)
            pos_relative_square = torch.sum(
                pos_relative**2, dim=-1
            ) + 2 * self.distance**2 * torch.eye(pos.shape[1])
            # add 2*self.distance**2 * torch.eye(self.dim) to remove self loop
            graph_list.append(
                torch.stack(
                    torch.where(pos_relative_square < self.distance**2),
                )
            )
        return graph_list


class StateBasedGraphDataset(Dataset):
    """
    Generate dataset with graph generator"""

    def __init__(
        self,
        data: ndarray,
        graph_generator: Callable[[ndarray], list[ndarray]],
        dt: float = 0.01,
    ):
        super().__init__()
        data = data.astype(np.float32)
        data_list = []
        target_list = []
        for x in data:
            data_list.append(torch.from_numpy(x[:-1]))
            target_list.append(torch.from_numpy(x[1:] - x[:-1]) / dt)
        self.x = torch.cat(data_list, axis=0)
        self.t = torch.cat(target_list, axis=0)
        self.g = graph_generator(self.x)
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.g[index], self.t[index]


class StaticGraphDataset(Dataset):
    """
    Generate dataset with static graph"""

    def __init__(self, data: ndarray, graph: list, dt: float = 0.01):
        super().__init__()
        data = data.astype(np.float32)
        data_list = []
        target_list = []
        graph_list = []
        for i, g in enumerate(graph):
            data_list.append(torch.from_numpy(data[i, :-1]))
            target_list.append(torch.from_numpy(data[i, 1:] - data[i, :-1]) / dt)
            graph_list.extend([torch.tensor(g) for _ in range(data.shape[1] - 1)])
        self.x = torch.cat(data_list, axis=0)
        self.t = torch.cat(target_list, axis=0)
        self.g = graph_list
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.g[index], self.t[index]
