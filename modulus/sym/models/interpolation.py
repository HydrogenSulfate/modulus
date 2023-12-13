# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
from typing import Dict, Tuple, Callable, List, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import Tensor

# TODO enum causes segmentation faults with current torch script. Go back to enum after torch script update
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


def linear_step(x: Tensor) -> Tensor:
    return paddle.clip(x, 0, 1)


def smooth_step_1(x: Tensor) -> Tensor:
    return paddle.clip(3 * x**2 - 2 * x**3, 0, 1)


def smooth_step_2(x: Tensor) -> Tensor:
    return paddle.clip(x**3 * (6 * x**2 - 15 * x + 10), 0, 1)


def nearest_neighbor_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    return paddle.ones(dist_vec.shape[:-2] + [1] + [1])


def _hyper_cube_weighting(lower_point: Tensor, upper_point: Tensor) -> Tensor:
    dim = lower_point.shape[-1]
    weights = []
    weights = [upper_point[..., 0], lower_point[..., 0]]
    for i in range(1, dim):
        new_weights = []
        for w in weights:
            new_weights.append(w * upper_point[..., i])
            new_weights.append(w * lower_point[..., i])
        weights = new_weights
    weights = paddle.stack(weights, axis=-1)
    return paddle.unsqueeze(weights, axis=-1)


def linear_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    normalized_dist_vec = dist_vec / dx
    lower_point = normalized_dist_vec[..., 0, :]
    upper_point = -normalized_dist_vec[..., -1, :]
    return _hyper_cube_weighting(lower_point, upper_point)


def smooth_step_1_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    normalized_dist_vec = dist_vec / dx
    lower_point = smooth_step_1(normalized_dist_vec[..., 0, :])
    upper_point = smooth_step_1(-normalized_dist_vec[..., -1, :])
    return _hyper_cube_weighting(lower_point, upper_point)


def smooth_step_2_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    normalized_dist_vec = dist_vec / dx
    lower_point = smooth_step_2(normalized_dist_vec[..., 0, :])
    upper_point = smooth_step_2(-normalized_dist_vec[..., -1, :])
    return _hyper_cube_weighting(lower_point, upper_point)


def gaussian_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    dim = dx.shape[-1]
    sharpen = 2.0
    sigma = dx / sharpen
    factor = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * sigma.prod())
    gaussian = paddle.exp(-0.5 * paddle.square(dist_vec / sigma))
    gaussian = factor * gaussian.prod(axis=-1)
    norm = gaussian.sum(axis=2, keepdim=True)
    weights = paddle.unsqueeze(gaussian / norm, axis=3)
    return weights


def _gather_nd(params: Tensor, indices: Tensor) -> Tensor:
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
            f"the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}"
        )
    x = indices.reshape((num_samples, m))
    perm_4 = list(range(x.ndim))
    perm_4[0] = 1
    perm_4[1] = 0
    indices = x.transpose(perm=perm_4).tolist()
    output = params[indices]
    return output.reshape(out_shape)


def index_values_high_mem(points: Tensor, idx: Tensor) -> Tensor:
    idx = idx.unsqueeze(3).repeat_interleave(points.shape[-1], axis=3)
    points = points.unsqueeze(1).repeat_interleave(idx.shape[1], axis=1)
    out = paddle.take_along_axis(points, axis=2, indices=idx)
    return out


def index_values_low_mem(points: Tensor, idx: Tensor) -> Tensor:
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
    batch_indices = paddle.reshape(
        paddle.tile(
            paddle.unsqueeze(
                paddle.arange(0, batch_size).to(device),
                axis=0,
            ),
            (num_points * K,),
        ),
        [-1],
    )  # BNK
    point_indices = paddle.reshape(idx, [-1])  # BNK
    vertices = _gather_nd(
        points, paddle.stack((batch_indices, point_indices), axis=1)
    )  # BNKxC
    vertices4d = paddle.reshape(
        vertices, [batch_size, num_points, K, num_features]
    )  # BxNxKxC
    return vertices4d


def _grid_knn_idx(
    query_points: Tensor,
    grid: List[Tuple[float, float, int]],
    stride: int,
    padding: bool = True,
) -> Tensor:

    # set k
    k = stride // 2

    # set device
    device = query_points.place

    # find nearest neighbors of query points from a grid
    # dx vector on grid
    dx = paddle.to_tensor([((x[1] - x[0]) / (x[2] - 1)) for x in grid])
    dx = dx.reshape([1, 1, len(grid)]).to(device)

    # min point on grid (this will change if we are padding the grid)
    start = paddle.to_tensor([val[0] for val in grid]).to(device)
    if padding:
        start = start - k * dx
    start = start.reshape([1, 1, len(grid)])

    # this is the center nearest neighbor in the grid
    center_idx = ((query_points - start) / dx + stride / 2.0 % 1.0).to("int64")

    # index window
    idx_add = (
        paddle.arange(-((stride - 1) // 2), stride // 2 + 1)
        .reshape([1, 1, -1])
        .to(device)
    )

    # find all index in window around center index
    # TODO make for more general diminsions
    if len(grid) == 1:
        idx_row_0 = center_idx[..., 0:1] + idx_add
        idx = idx_row_0.reshape([idx_row_0.shape[0:2] + list([int(stride)])])
    elif len(grid) == 2:
        dim_size_1 = grid[1][2]
        if padding:
            dim_size_1 += 2 * k
        idx_row_0 = dim_size_1 * (center_idx[..., 0:1] + idx_add)
        idx_row_0 = idx_row_0.unsqueeze(-1)
        idx_row_1 = center_idx[..., 1:2] + idx_add
        idx_row_1 = idx_row_1.unsqueeze(2)
        idx = (idx_row_0 + idx_row_1).reshape(
            idx_row_0.shape[0:2] + list([int(stride**2)])
        )
    elif len(grid) == 3:
        dim_size_1 = grid[1][2]
        dim_size_2 = grid[2][2]
        if padding:
            dim_size_1 += 2 * k
            dim_size_2 += 2 * k
        idx_row_0 = dim_size_2 * dim_size_1 * (center_idx[..., 0:1] + idx_add)
        idx_row_0 = idx_row_0.unsqueeze(-1).unsqueeze(-1)
        idx_row_1 = dim_size_2 * (center_idx[..., 1:2] + idx_add)
        idx_row_1 = idx_row_1.unsqueeze(2).unsqueeze(-1)
        idx_row_2 = center_idx[..., 2:3] + idx_add
        idx_row_2 = idx_row_2.unsqueeze(2).unsqueeze(3)
        idx = (idx_row_0 + idx_row_1 + idx_row_2).reshape(
            idx_row_0.shape[0:2] + list([int(stride**3)])
        )
    else:
        raise RuntimeError

    return idx


# TODO currently the `tolist` operation is not supported by torch script and when fixed torch script will be used
def interpolation(
    query_points: Tensor,
    context_grid: Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str = "smooth_step_2",
    mem_speed_trade: bool = True,
) -> Tensor:

    # set stride TODO this will be replaced with InterpolationType later
    if interpolation_type == "nearest_neighbor":
        stride = 1
    elif interpolation_type == "linear":
        stride = 2
    elif interpolation_type == "smooth_step_1":
        stride = 2
    elif interpolation_type == "smooth_step_2":
        stride = 2
    elif interpolation_type == "gaussian":
        stride = 5
    else:
        raise RuntimeError

    # set device
    device = query_points.place
    dims = len(grid)
    nr_channels = context_grid.shape[0]
    dx = [((x[1] - x[0]) / (x[2] - 1)) for x in grid]

    # generate mesh grid of position information [grid_dim_1, grid_dim_2, ..., 2-3]
    # NOTE the mesh grid is padded by stride//2
    k = stride // 2
    linspace = [
        paddle.linspace(x[0] - k * dx_i, stop=x[1] + k * dx_i, num=x[2] + 2 * k)
        for x, dx_i in zip(grid, dx)
    ]
    meshgrid = paddle.meshgrid(linspace)
    meshgrid = paddle.stack(meshgrid, axis=-1).to(device)

    # pad context grid by k to avoid cuts on corners
    padding = dims * (k, k)
    context_grid = F.pad(context_grid, padding)

    # reshape query points, context grid and mesh grid for easier indexing
    # [1, grid_dim_1*grid_dim_2*..., 2-4]
    nr_grid_points = int(paddle.to_tensor([(x[2] + 2 * k) for x in grid]).prod())
    meshgrid = meshgrid.reshape([1, nr_grid_points, dims])
    context_grid = paddle.reshape(context_grid, [1, nr_channels, nr_grid_points])

    x = context_grid
    perm_5 = list(range(x.ndim))
    perm_5[1] = 2
    perm_5[2] = 1
    context_grid = paddle.transpose(x, perm=perm_5)
    query_points = query_points.unsqueeze(0)

    # compute index of nearest neighbor on grid to query points
    idx = _grid_knn_idx(query_points, grid, stride, padding=True)

    # index mesh grid to get distance vector
    if mem_speed_trade:
        mesh_grid_idx = index_values_low_mem(meshgrid, idx)
    else:
        mesh_grid_idx = index_values_high_mem(meshgrid, idx)
    dist_vec = query_points.unsqueeze(2) - mesh_grid_idx

    # make tf dx vec (for interpolation function)
    dx = paddle.to_tensor(dx, dtype="float32")
    dx = paddle.reshape(dx, [1, 1, 1, dims]).to(device)

    # compute bump function
    if interpolation_type == "nearest_neighbor":
        weights = nearest_neighbor_weighting(dist_vec, dx)
    elif interpolation_type == "linear":
        weights = linear_weighting(dist_vec, dx)
    elif interpolation_type == "smooth_step_1":
        weights = smooth_step_1_weighting(dist_vec, dx)
    elif interpolation_type == "smooth_step_2":
        weights = smooth_step_2_weighting(dist_vec, dx)
    elif interpolation_type == "gaussian":
        weights = gaussian_weighting(dist_vec, dx)
    else:
        raise RuntimeError

    # index context grid with index
    if mem_speed_trade:
        context_grid_idx = index_values_low_mem(context_grid, idx)
    else:
        context_grid_idx = index_values_high_mem(context_grid, idx)

    # interpolate points
    product = weights * context_grid_idx
    interpolated_points = product.sum(axis=2)

    return interpolated_points[0]
