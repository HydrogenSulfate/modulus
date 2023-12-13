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

import paddle
from modulus.sym.models.interpolation import interpolation
import numpy as np


def test_interpolation():
    device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"
    grid = [(-1, 2, 30), (-1, 2, 30), (-1, 2, 30)]
    np_linspace = [np.linspace(x[0], x[1], x[2]) for x in grid]
    np_mesh_grid = np.meshgrid(*np_linspace, indexing="ij")
    np_mesh_grid = np.stack(np_mesh_grid, axis=0)
    mesh_grid = paddle.to_tensor(data=np_mesh_grid, dtype="float32").to(device)
    sin_grid = paddle.sin(
        x=mesh_grid[0:1, :, :] + mesh_grid[1:2, :, :] ** 2 + mesh_grid[2:3, :, :] ** 3
    ).to(device)
    nr_points = 100
    out_28 = paddle.stack(
        x=[
            paddle.linspace(start=0.0, stop=1.0, num=nr_points),
            paddle.linspace(start=0.0, stop=1.0, num=nr_points),
            paddle.linspace(start=0.0, stop=1.0, num=nr_points),
        ],
        axis=-1,
    ).to(device)
    out_28.stop_gradient = not True
    query_points = out_28
    interpolation_types = [
        "nearest_neighbor",
        "linear",
        "smooth_step_1",
        "smooth_step_2",
        "gaussian",
    ]
    for i_type in interpolation_types:
        computed_interpolation = interpolation(
            query_points,
            sin_grid,
            grid=grid,
            interpolation_type=i_type,
            mem_speed_trade=False,
        )
        np_computed_interpolation = computed_interpolation.cpu().detach().numpy()
        np_ground_truth = (
            paddle.sin(
                x=query_points[:, 0:1]
                + query_points[:, 1:2] ** 2
                + query_points[:, 2:3] ** 3
            )
            .cpu()
            .detach()
            .numpy()
        )
        difference = np.linalg.norm(
            (np_computed_interpolation - np_ground_truth) / nr_points
        )
        assert difference < 0.01, "Test failed!"
