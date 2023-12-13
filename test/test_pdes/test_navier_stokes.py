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
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
import numpy as np
import os


def test_navier_stokes_equation():
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    u = np.exp(2 * x + y + z + t)
    v = np.exp(x + 2 * y + z + t)
    w = np.exp(x + y + 2 * z + t)
    p = np.exp(x + y + z + t)
    rho = 1.0
    nu = 0.2
    u__t = 1 * np.exp(2 * x + y + z + t)
    u__x = 2 * np.exp(2 * x + y + z + t)
    u__y = 1 * np.exp(2 * x + y + z + t)
    u__z = 1 * np.exp(2 * x + y + z + t)
    u__x__x = 2 * 2 * np.exp(2 * x + y + z + t)
    u__y__y = 1 * 1 * np.exp(2 * x + y + z + t)
    u__z__z = 1 * 1 * np.exp(2 * x + y + z + t)
    u__x__y = 1 * 2 * np.exp(2 * x + y + z + t)
    u__x__z = 1 * 2 * np.exp(2 * x + y + z + t)
    u__y__z = 1 * 1 * np.exp(2 * x + y + z + t)
    u__y__x = u__x__y
    u__z__x = u__x__z
    u__z__y = u__y__z
    v__t = 1 * np.exp(x + 2 * y + z + t)
    v__x = 1 * np.exp(x + 2 * y + z + t)
    v__y = 2 * np.exp(x + 2 * y + z + t)
    v__z = 1 * np.exp(x + 2 * y + z + t)
    v__x__x = 1 * 1 * np.exp(x + 2 * y + z + t)
    v__y__y = 2 * 2 * np.exp(x + 2 * y + z + t)
    v__z__z = 1 * 1 * np.exp(x + 2 * y + z + t)
    v__x__y = 2 * 1 * np.exp(x + 2 * y + z + t)
    v__x__z = 1 * 1 * np.exp(x + 2 * y + z + t)
    v__y__z = 1 * 2 * np.exp(x + 2 * y + z + t)
    v__y__x = v__x__y
    v__z__x = v__x__z
    v__z__y = v__y__z
    w__t = 1 * np.exp(x + y + 2 * z + t)
    w__x = 1 * np.exp(x + y + 2 * z + t)
    w__y = 1 * np.exp(x + y + 2 * z + t)
    w__z = 2 * np.exp(x + y + 2 * z + t)
    w__x__x = 1 * 1 * np.exp(x + y + 2 * z + t)
    w__y__y = 1 * 1 * np.exp(x + y + 2 * z + t)
    w__z__z = 2 * 2 * np.exp(x + y + 2 * z + t)
    w__x__y = 1 * 1 * np.exp(x + y + 2 * z + t)
    w__x__z = 2 * 1 * np.exp(x + y + 2 * z + t)
    w__y__z = 2 * 1 * np.exp(x + y + 2 * z + t)
    w__y__x = w__x__y
    w__z__x = w__x__z
    w__z__y = w__y__z
    p__x = 1 * np.exp(x + y + z + t)
    p__y = 1 * np.exp(x + y + z + t)
    p__z = 1 * np.exp(x + y + z + t)
    continuity_equation_true = 0 + rho * u__x + rho * v__y + rho * w__z
    momentum_x_equation_true = (
        rho * u__t
        + u * rho * u__x
        + v * rho * u__y
        + w * rho * u__z
        + p__x
        - rho * nu * u__x__x
        - rho * nu * u__y__y
        - rho * nu * u__z__z
    )
    momentum_y_equation_true = (
        rho * v__t
        + u * rho * v__x
        + v * rho * v__y
        + w * rho * v__z
        + p__y
        - rho * nu * v__x__x
        - rho * nu * v__y__y
        - rho * nu * v__z__z
    )
    momentum_z_equation_true = (
        rho * w__t
        + u * rho * w__x
        + v * rho * w__y
        + w * rho * w__z
        + p__z
        - rho * nu * w__x__x
        - rho * nu * w__y__y
        - rho * nu * w__z__z
    )
    navier_stokes_eq = NavierStokes(nu=nu, rho=rho, dim=3, time=True)
    evaluations_continuity = navier_stokes_eq.make_nodes()[0].evaluate(
        {
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "w__z": paddle.to_tensor(data=w__z, dtype="float32"),
        }
    )
    evaluations_momentum_x = navier_stokes_eq.make_nodes()[1].evaluate(
        {
            "u__t": paddle.to_tensor(data=u__t, dtype="float32"),
            "u__x": paddle.to_tensor(data=u__x, dtype="float32"),
            "u__y": paddle.to_tensor(data=u__y, dtype="float32"),
            "u__z": paddle.to_tensor(data=u__z, dtype="float32"),
            "u__x__x": paddle.to_tensor(data=u__x__x, dtype="float32"),
            "u__y__y": paddle.to_tensor(data=u__y__y, dtype="float32"),
            "u__z__z": paddle.to_tensor(data=u__z__z, dtype="float32"),
            "p__x": paddle.to_tensor(data=p__x, dtype="float32"),
            "u": paddle.to_tensor(data=u, dtype="float32"),
            "v": paddle.to_tensor(data=v, dtype="float32"),
            "w": paddle.to_tensor(data=w, dtype="float32"),
        }
    )
    evaluations_momentum_y = navier_stokes_eq.make_nodes()[2].evaluate(
        {
            "v__t": paddle.to_tensor(data=v__t, dtype="float32"),
            "v__x": paddle.to_tensor(data=v__x, dtype="float32"),
            "v__y": paddle.to_tensor(data=v__y, dtype="float32"),
            "v__z": paddle.to_tensor(data=v__z, dtype="float32"),
            "v__x__x": paddle.to_tensor(data=v__x__x, dtype="float32"),
            "v__y__y": paddle.to_tensor(data=v__y__y, dtype="float32"),
            "v__z__z": paddle.to_tensor(data=v__z__z, dtype="float32"),
            "p__y": paddle.to_tensor(data=p__y, dtype="float32"),
            "u": paddle.to_tensor(data=u, dtype="float32"),
            "v": paddle.to_tensor(data=v, dtype="float32"),
            "w": paddle.to_tensor(data=w, dtype="float32"),
        }
    )
    evaluations_momentum_z = navier_stokes_eq.make_nodes()[3].evaluate(
        {
            "w__t": paddle.to_tensor(data=w__t, dtype="float32"),
            "w__x": paddle.to_tensor(data=w__x, dtype="float32"),
            "w__y": paddle.to_tensor(data=w__y, dtype="float32"),
            "w__z": paddle.to_tensor(data=w__z, dtype="float32"),
            "w__x__x": paddle.to_tensor(data=w__x__x, dtype="float32"),
            "w__y__y": paddle.to_tensor(data=w__y__y, dtype="float32"),
            "w__z__z": paddle.to_tensor(data=w__z__z, dtype="float32"),
            "p__z": paddle.to_tensor(data=p__z, dtype="float32"),
            "u": paddle.to_tensor(data=u, dtype="float32"),
            "v": paddle.to_tensor(data=v, dtype="float32"),
            "w": paddle.to_tensor(data=w, dtype="float32"),
        }
    )
    continuity_eq_eval_pred = evaluations_continuity["continuity"].numpy()
    momentum_x_eq_eval_pred = evaluations_momentum_x["momentum_x"].numpy()
    momentum_y_eq_eval_pred = evaluations_momentum_y["momentum_y"].numpy()
    momentum_z_eq_eval_pred = evaluations_momentum_z["momentum_z"].numpy()
    assert np.allclose(
        continuity_eq_eval_pred, continuity_equation_true
    ), "Test Failed!"
    assert np.allclose(
        momentum_x_eq_eval_pred, momentum_x_equation_true
    ), "Test Failed!"
    assert np.allclose(
        momentum_y_eq_eval_pred, momentum_y_equation_true
    ), "Test Failed!"
    assert np.allclose(
        momentum_z_eq_eval_pred, momentum_z_equation_true
    ), "Test Failed!"


if __name__ == "__main__":
    test_navier_stokes_equation()
