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

import torch
import pytest
import random

from modulus.models.layers.activations import Identity, Stan, SquarePlus
from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_identity(device):
    """Test identity function in layers"""
    func = Identity().to(device)
    # Random tensor of random size
    tensor_dim = random.randint(1, 5)
    tensor_size = torch.randint(low=1, high=8, size=(tensor_dim,)).tolist()
    invar = torch.randn(*tensor_size, device=device)

    outvar = func(invar)
    assert common.compare_output(invar, outvar)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_stan(device):
    """Test Stan function in layers"""
    func = Stan(out_features=2).to(device)
    # Doc string example handles accuracy
    bsize = random.randint(1, 8)
    invar = torch.randn(bsize, 2).to(device)
    outvar = func(invar)
    # Learnable param should be 1.0 init
    tarvar = (invar + 1) * torch.tanh(invar)
    assert common.compare_output(tarvar, outvar)

    # Also test failure case
    try:
        func = Stan(out_features=random.randint(1, 3)).to(device)
        invar = torch.randn(2, 4).to(device)
        outvar = func(invar)
        assert False, "Failed to error for invalid input feature dimension"
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_activation_squareplus(device):
    """Test square plus function in layers"""
    func = SquarePlus().to(device)
    func.b = 0
    # Ones tensor of random size
    tensor_dim = random.randint(1, 3)
    tensor_size = torch.randint(low=1, high=4, size=(tensor_dim,)).tolist()
    invar = torch.ones(*tensor_size, device=device)

    outvar = func(invar)
    assert common.compare_output(torch.ones_like(invar), outvar)
