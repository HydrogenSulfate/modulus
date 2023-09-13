import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
import enum
import math
from typing import Dict, Tuple, Callable, List, Union
import numpy as np
"""
@enum.unique
class InterpolationType(enum.Enum):
    NEAREST_NEIGHBOR = (1, 1)
    LINEAR = (2, 2)
    SMOOTH_STEP_1 = (3, 2)
    SMOOTH_STEP_2 = (4, 2)
    GAUSSIAN = (6, 5)

    def __init__(self, index, stride):
        self.index = index
        self.stride = stride
"""


>>>@torch.jit.script
def linear_step(x: paddle.Tensor) ->paddle.Tensor:
    return paddle.clip(x=x, min=0, max=1)


>>>@torch.jit.script
def smooth_step_1(x: paddle.Tensor) ->paddle.Tensor:
    return paddle.clip(x=3 * x ** 2 - 2 * x ** 3, min=0, max=1)


>>>@torch.jit.script
def smooth_step_2(x: paddle.Tensor) ->paddle.Tensor:
    return paddle.clip(x=x ** 3 * (6 * x ** 2 - 15 * x + 10), min=0, max=1)


>>>@torch.jit.script
def nearest_neighbor_weighting(dist_vec: paddle.Tensor, dx: paddle.Tensor
    ) ->paddle.Tensor:
    return paddle.ones(shape=dist_vec.shape[:-2] + [1] + [1])


>>>@torch.jit.script
def _hyper_cube_weighting(lower_point: paddle.Tensor, upper_point: paddle.
    Tensor) ->paddle.Tensor:
    dim = lower_point.shape[-1]
    weights = []
    weights = [upper_point[..., 0], lower_point[..., 0]]
    for i in range(1, dim):
        new_weights = []
        for w in weights:
            new_weights.append(w * upper_point[..., i])
            new_weights.append(w * lower_point[..., i])
        weights = new_weights
    weights = paddle.stack(x=weights, axis=-1)
    return paddle.unsqueeze(x=weights, axis=-1)


>>>@torch.jit.script
def linear_weighting(dist_vec: paddle.Tensor, dx: paddle.Tensor
    ) ->paddle.Tensor:
    normalized_dist_vec = dist_vec / dx
    lower_point = normalized_dist_vec[..., 0, :]
    upper_point = -normalized_dist_vec[..., -1, :]
    return _hyper_cube_weighting(lower_point, upper_point)


>>>@torch.jit.script
def smooth_step_1_weighting(dist_vec: paddle.Tensor, dx: paddle.Tensor
    ) ->paddle.Tensor:
    normalized_dist_vec = dist_vec / dx
    lower_point = smooth_step_1(normalized_dist_vec[..., 0, :])
    upper_point = smooth_step_1(-normalized_dist_vec[..., -1, :])
    return _hyper_cube_weighting(lower_point, upper_point)


>>>@torch.jit.script
def smooth_step_2_weighting(dist_vec: paddle.Tensor, dx: paddle.Tensor
    ) ->paddle.Tensor:
    normalized_dist_vec = dist_vec / dx
    lower_point = smooth_step_2(normalized_dist_vec[..., 0, :])
    upper_point = smooth_step_2(-normalized_dist_vec[..., -1, :])
    return _hyper_cube_weighting(lower_point, upper_point)


>>>@torch.jit.script
def gaussian_weighting(dist_vec: paddle.Tensor, dx: paddle.Tensor
    ) ->paddle.Tensor:
    dim = dx.shape[-1]
    sharpen = 2.0
    sigma = dx / sharpen
    factor = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * sigma.prod())
    gaussian = paddle.exp(x=-0.5 * paddle.square(x=dist_vec / sigma))
    gaussian = factor * gaussian.prod(axis=-1)
    norm = gaussian.sum(axis=2, keepdim=True)
    weights = paddle.unsqueeze(x=gaussian / norm, axis=3)
    return weights


def _gather_nd(params: paddle.Tensor, indices: paddle.Tensor) ->paddle.Tensor:
    """As seen here https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/30"""
    orig_shape = list(indices.shape)
    num_samples = 1
    for s in orig_shape[:-1]:
        num_samples *= s
    m = orig_shape[-1]
    n = len(params.shape)
    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
            )
    x = indices.reshape((num_samples, m))
    perm_4 = list(range(x.ndim))
    perm_4[0] = 1
    perm_4[1] = 0
    indices = x.transpose(perm=perm_4).tolist()
    output = params[indices]
    return output.reshape(out_shape)


>>>@torch.jit.script
def index_values_high_mem(points: paddle.Tensor, idx: paddle.Tensor
    ) ->paddle.Tensor:
    idx = idx.unsqueeze(axis=3).repeat_interleave(repeats=points.shape[-1],
        axis=3)
    points = points.unsqueeze(axis=1).repeat_interleave(repeats=idx.shape[1
        ], axis=1)
    out = paddle.take_along_axis(arr=points, axis=2, indices=idx)
    return out


def index_values_low_mem(points: paddle.Tensor, idx: paddle.Tensor
    ) ->paddle.Tensor:
    """
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
    Output:
        out: (b,m,n,c) float32 array, interpolated point values
    """
    device = points.place
    idxShape = idx.shape
    batch_size = idxShape[0]
    num_points = idxShape[1]
    K = idxShape[2]
    num_features = points.shape[2]
    batch_indices = paddle.reshape(x=paddle.tile(x=paddle.unsqueeze(x=
        paddle.arange(start=0, end=batch_size).to(device), axis=0),
        repeat_times=(num_points * K,)), shape=[-1])
    point_indices = paddle.reshape(x=idx, shape=[-1])
    vertices = _gather_nd(points, paddle.stack(x=(batch_indices,
        point_indices), axis=1))
    vertices4d = paddle.reshape(x=vertices, shape=[batch_size, num_points,
        K, num_features])
    return vertices4d


>>>@torch.jit.script
def _grid_knn_idx(query_points: paddle.Tensor, grid: List[Tuple[float,
    float, int]], stride: int, padding: bool=True) ->paddle.Tensor:
    k = stride // 2
    device = query_points.place
    dx = paddle.to_tensor(data=[((x[1] - x[0]) / (x[2] - 1)) for x in grid])
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    dx = dx.reshape([1, 1, len(grid)]).to(device)
    start = paddle.to_tensor(data=[val[0] for val in grid]).to(device)
    if padding:
        start = start - k * dx
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    start = start.reshape([1, 1, len(grid)])
    center_idx = ((query_points - start) / dx + stride / 2.0 % 1.0).to('int64')
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    idx_add = paddle.arange(start=-((stride - 1) // 2), end=stride // 2 + 1
        ).reshape([1, 1, -1]).to(device)
    if len(grid) == 1:
        idx_row_0 = center_idx[..., 0:1] + idx_add
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        idx = idx_row_0.reshape([idx_row_0.shape[0:2] + list([int(stride)])])
    elif len(grid) == 2:
        dim_size_1 = grid[1][2]
        if padding:
            dim_size_1 += 2 * k
        idx_row_0 = dim_size_1 * (center_idx[..., 0:1] + idx_add)
        idx_row_0 = idx_row_0.unsqueeze(axis=-1)
        idx_row_1 = center_idx[..., 1:2] + idx_add
        idx_row_1 = idx_row_1.unsqueeze(axis=2)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        idx = (idx_row_0 + idx_row_1).reshape(idx_row_0.shape[0:2] + list([int
            (stride ** 2)]))
    elif len(grid) == 3:
        dim_size_1 = grid[1][2]
        dim_size_2 = grid[2][2]
        if padding:
            dim_size_1 += 2 * k
            dim_size_2 += 2 * k
        idx_row_0 = dim_size_2 * dim_size_1 * (center_idx[..., 0:1] + idx_add)
        idx_row_0 = idx_row_0.unsqueeze(axis=-1).unsqueeze(axis=-1)
        idx_row_1 = dim_size_2 * (center_idx[..., 1:2] + idx_add)
        idx_row_1 = idx_row_1.unsqueeze(axis=2).unsqueeze(axis=-1)
        idx_row_2 = center_idx[..., 2:3] + idx_add
        idx_row_2 = idx_row_2.unsqueeze(axis=2).unsqueeze(axis=3)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        idx = (idx_row_0 + idx_row_1 + idx_row_2).reshape(idx_row_0.shape[0:2] +
            list([int(stride ** 3)]))
    else:
        raise RuntimeError
    return idx


def interpolation(query_points: paddle.Tensor, context_grid: paddle.Tensor,
    grid: List[Tuple[float, float, int]], interpolation_type: str=
    'smooth_step_2', mem_speed_trade: bool=True) ->paddle.Tensor:
    if interpolation_type == 'nearest_neighbor':
        stride = 1
    elif interpolation_type == 'linear':
        stride = 2
    elif interpolation_type == 'smooth_step_1':
        stride = 2
    elif interpolation_type == 'smooth_step_2':
        stride = 2
    elif interpolation_type == 'gaussian':
        stride = 5
    else:
        raise RuntimeError
    device = query_points.place
    dims = len(grid)
    nr_channels = context_grid.shape[0]
    dx = [((x[1] - x[0]) / (x[2] - 1)) for x in grid]
    k = stride // 2
    linspace = [paddle.linspace(start=x[0] - k * dx_i, stop=x[1] + k * dx_i,
        num=x[2] + 2 * k) for x, dx_i in zip(grid, dx)]
    meshgrid = paddle.meshgrid(linspace)
    meshgrid = paddle.stack(x=meshgrid, axis=-1).to(device)
    padding = dims * (k, k)
    context_grid = paddle.nn.functional.pad(context_grid, padding)
    nr_grid_points = int(paddle.to_tensor(data=[(x[2] + 2 * k) for x in
        grid]).prod())
    """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
    meshgrid = meshgrid.reshape([1, nr_grid_points, dims])
    context_grid = paddle.reshape(x=context_grid, shape=[1, nr_channels,
        nr_grid_points])
    x = context_grid
    perm_5 = list(range(x.ndim))
    perm_5[1] = 2
    perm_5[2] = 1
    context_grid = paddle.transpose(x=x, perm=perm_5)
    query_points = query_points.unsqueeze(axis=0)
    idx = _grid_knn_idx(query_points, grid, stride, padding=True)
    if mem_speed_trade:
        mesh_grid_idx = index_values_low_mem(meshgrid, idx)
    else:
        mesh_grid_idx = index_values_high_mem(meshgrid, idx)
    dist_vec = query_points.unsqueeze(axis=2) - mesh_grid_idx
    dx = paddle.to_tensor(data=dx, dtype='float32')
    dx = paddle.reshape(x=dx, shape=[1, 1, 1, dims]).to(device)
    if interpolation_type == 'nearest_neighbor':
        weights = nearest_neighbor_weighting(dist_vec, dx)
    elif interpolation_type == 'linear':
        weights = linear_weighting(dist_vec, dx)
    elif interpolation_type == 'smooth_step_1':
        weights = smooth_step_1_weighting(dist_vec, dx)
    elif interpolation_type == 'smooth_step_2':
        weights = smooth_step_2_weighting(dist_vec, dx)
    elif interpolation_type == 'gaussian':
        weights = gaussian_weighting(dist_vec, dx)
    else:
        raise RuntimeError
    if mem_speed_trade:
        context_grid_idx = index_values_low_mem(context_grid, idx)
    else:
        context_grid_idx = index_values_high_mem(context_grid, idx)
    product = weights * context_grid_idx
    interpolated_points = product.sum(axis=2)
    return interpolated_points[0]
