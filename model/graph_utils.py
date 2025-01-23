import torch
from torch import Tensor


def non_zero_div(x: Tensor, y: Tensor) -> Tensor:
    """
    treat the division by zero as 0, counterpart of np.devide(x, y, out=np.zeros_like(x), where=y!=0)
    """
    mask = y != 0.0
    mask = mask.squeeze(-1)
    out = torch.zeros_like(x)
    out[mask] = x[mask] / y[mask]
    return out


def unsorted_segment_sum(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    """
    length of segment_ids should be the same as the first dimension of data.
    Notes:
    .. math:: x=y
    """
    device = data.device
    data, segment_ids = data.to(device), segment_ids.to(device)
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    device = data.device
    data, segment_ids = data.to(device), segment_ids.to(device)
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0).to(device)
    count = data.new_full(result_shape, 0).to(device)
    result.scatter_add_(0, segment_ids, data).to(device)
    count.scatter_add_(0, segment_ids, torch.ones_like(data)).to(device)
    return result / count.clamp(min=1)
