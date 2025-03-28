import torch
from torch import nn, Tensor, optim, autograd
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

import numpy as np

import logging

from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Trainer:
    """
    A class to train a model with a given criterion and regularizers.
    """

    def __init__(
        self,
        criterion: Module,
        regularizers: list[Module] = [],
    ):
        self.criterion = criterion
        self.regularizers = regularizers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        model: Module,
        dataset: Dataset,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int,
        require_grad: bool = True,
        varidation_dataset: Optional[Dataset] = None,
        verbose: bool = True,
        log_path: Optional[str] = None,
    ):
        history = {"train_loss": [], "train_reg": [], "train_sum": [], "val_loss": []}
        best_loss = float("inf")

        model.to(self.device)

        val_loader = (
            None
            if varidation_dataset is None
            else DataLoader(
                varidation_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.graph_collate_fn,
            )
        )
        for epoch in range(epochs):
            model.train()
            train_loss_display = 0
            train_reg_display = 0
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=self.graph_collate_fn,
            )
            for x, g, t in data_loader:
                x, g, t = x.to(self.device), g.to(self.device), t.to(self.device)
                if require_grad:
                    x.requires_grad = True
                optimizer.zero_grad()
                y = model(x, g)
                loss = self.criterion(y, t)
                train_loss_display += loss.item()
                for regularizer in self.regularizers:
                    reg_loss = regularizer(model)
                    train_reg_display += reg_loss.item()
                    loss += reg_loss
                loss.backward()
                optimizer.step()
            train_loss_display /= len(data_loader)
            train_reg_display /= len(data_loader)
            history["train_loss"].append(train_loss_display)
            history["train_reg"].append(train_reg_display)
            history["train_sum"].append(train_loss_display + train_reg_display)
            val_loss_display = 0
            if val_loader is not None:
                model.eval()
                for x, g, t in val_loader:
                    x, g, t = x.to(self.device), g.to(self.device), t.to(self.device)
                    if require_grad:
                        x.requires_grad = True
                    y = model(x, g)
                    loss = self.criterion(y, t)
                    val_loss_display += loss.item()
                val_loss_display /= len(val_loader)
                if log_path is not None:
                    if val_loss_display < best_loss:
                        best_loss = val_loss_display
                        torch.save(model.state_dict(), f"{log_path}/best_model.pth")
            history["val_loss"].append(val_loss_display)
            if log_path is not None:
                np.save(f"{log_path}/history.npy", history, allow_pickle=True)
                torch.save(model.state_dict(), f"{log_path}/final_model.pth")
            msg = f"Epoch: {epoch+1}/{epochs}, train_loss: {train_loss_display}"
            if train_reg_display > 0:
                msg += f", train_reg: {train_reg_display}, train_sum: {train_loss_display + train_reg_display}"
            if val_loss_display > 0:
                msg += f", val_loss: {val_loss_display}"
            if verbose:
                logger.info(msg)
            else:
                logger.debug(msg)
        return history

    @staticmethod
    def graph_collate_fn(batch):
        x, g, t = zip(*batch)
        x = torch.stack(x)
        t = torch.stack(t)
        n = x.size(1)
        g = [graph + bi * n for bi, graph in enumerate(g)]
        g = torch.cat(g, dim=-1)
        return x, g, t


class L1Reguralizer(Module):
    def __init__(
        self, cofficient_weight: float, cofficient_bias: Optional[float] = None
    ):
        super().__init__()
        self.cofficient_weight = cofficient_weight
        self.cofficient_bias = (
            cofficient_weight if cofficient_bias is None else cofficient_bias
        )

    def forward(self, model: Module) -> Tensor:
        params = model.parameters()
        params = [param for param in params]
        reg_loss = 0.0
        weight = params[0::2]
        bias = params[1::2]
        for w in weight:
            reg_loss += self.cofficient_weight * torch.sum(torch.abs(w))
        for b in bias:
            reg_loss += self.cofficient_bias * torch.sum(torch.abs(b))
        return reg_loss


class L2Reguralizer(Module):
    def __init__(
        self, cofficient_weight: float, cofficient_bias: Optional[float] = None
    ):
        super().__init__()
        self.cofficient_weight = cofficient_weight
        self.cofficient_bias = (
            cofficient_weight if cofficient_bias is None else cofficient_bias
        )

    def forward(self, model: Module) -> Tensor:
        params = model.parameters()
        params = [param for param in params]
        reg_loss = 0.0
        weight = params[0::2]
        bias = params[1::2]
        for w in weight:
            reg_loss += self.cofficient_weight * torch.sum(w**2)
        for b in bias:
            reg_loss += self.cofficient_bias * torch.sum(b**2)
        return reg_loss
