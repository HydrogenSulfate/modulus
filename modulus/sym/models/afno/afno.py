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
from functools import partial
from typing import Dict, List, Tuple
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(
            in_features=hidden_features, out_features=out_features
        )
        self.drop = paddle.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):
        super().__init__()
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        out_78 = paddle.create_parameter(
            shape=(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size,
                        self.block_size * self.hidden_size_factor,
                    ]
                )
            ).shape,
            dtype=(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size,
                        self.block_size * self.hidden_size_factor,
                    ]
                )
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size,
                        self.block_size * self.hidden_size_factor,
                    ]
                )
            ),
        )
        out_78.stop_gradient = not True
        self.w1 = out_78
        out_79 = paddle.create_parameter(
            shape=(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                    ]
                )
            ).shape,
            dtype=(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                    ]
                )
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                    ]
                )
            ),
        )
        out_79.stop_gradient = not True
        self.b1 = out_79
        out_80 = paddle.create_parameter(
            shape=(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                        self.block_size,
                    ]
                )
            ).shape,
            dtype=(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                        self.block_size,
                    ]
                )
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                self.scale
                * paddle.randn(
                    shape=[
                        2,
                        self.num_blocks,
                        self.block_size * self.hidden_size_factor,
                        self.block_size,
                    ]
                )
            ),
        )
        out_80.stop_gradient = not True
        self.w2 = out_80
        out_81 = paddle.create_parameter(
            shape=(
                self.scale * paddle.randn(shape=[2, self.num_blocks, self.block_size])
            ).shape,
            dtype=(
                self.scale * paddle.randn(shape=[2, self.num_blocks, self.block_size])
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                self.scale * paddle.randn(shape=[2, self.num_blocks, self.block_size])
            ),
        )
        out_81.stop_gradient = not True
        self.b2 = out_81

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.astype(dtype="float32")
        B, H, W, C = x.shape
        x = paddle.fft.rfft2(x=x, axes=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        o1_real = paddle.zeros(
            shape=[
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ]
        )
        o1_imag = paddle.zeros(
            shape=[
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ]
        )
        o2_real = paddle.zeros(shape=x.shape)
        o2_imag = paddle.zeros(shape=x.shape)
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        o1_real[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = paddle.nn.functional.relu(
            x=paddle.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].real(),
                self.w1[0],
            )
            - paddle.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].imag(),
                self.w1[1],
            )
            + self.b1[0]
        )
        o1_imag[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = paddle.nn.functional.relu(
            x=paddle.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].imag(),
                self.w1[0],
            )
            + paddle.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].real(),
                self.w1[1],
            )
            + self.b1[1]
        )
        o2_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            paddle.einsum(
                "...bi,bio->...bo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - paddle.einsum(
                "...bi,bio->...bo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )
        o2_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            paddle.einsum(
                "...bi,bio->...bo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + paddle.einsum(
                "...bi,bio->...bo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )
        x = paddle.stack(x=[o2_real, o2_imag], axis=-1)
        x = paddle.nn.functional.softshrink(x=x, threshold=self.sparsity_threshold)
        x = paddle.as_complex(x=x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = paddle.fft.irfft2(x=x, s=(H, W), axes=(1, 2), norm="ortho")
        x = x.astype(dtype)
        return x + bias


class Block(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(
            dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class AFNONet(paddle.nn.Layer):
    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=(16, 16),
        in_channels=1,
        out_channels=1,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ) -> None:
        super().__init__()
        assert (
            img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), f"img_size {img_size} should be divisible by patch_size {patch_size}"
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(paddle.nn.LayerNorm, eps=1e-06)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        out_82 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, num_patches, embed_dim]).shape,
            dtype=paddle.zeros(shape=[1, num_patches, embed_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=[1, num_patches, embed_dim])
            ),
        )
        out_82.stop_gradient = not True
        self.pos_embed = out_82
        self.pos_drop = paddle.nn.Dropout(drop_rate)
        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )
        self.head = paddle.nn.Linear(
            in_features=embed_dim,
            out_features=self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias_attr=False,
        )
        paddle.nn.initializer.TruncNormal(std=0.02)(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            paddle.nn.initializer.TruncNormal(std=0.02)(m.weight)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(1.0)
            init_Constant(m.weight)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        out = x.reshape(
            list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1]
        )
        out = paddle.transpose(x=out, perm=(0, 5, 1, 3, 2, 4))
        out = out.reshape(list(out.shape[:2]) + [self.img_size[0], self.img_size[1]])
        return out


class PatchEmbed(paddle.nn.Layer):
    def __init__(
        self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768
    ):
        super().__init__()
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = paddle.nn.Conv2D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_axis=2)
        perm_6 = list(range(x.ndim))
        perm_6[1] = 2
        perm_6[2] = 1
        x = x.transpose(perm=perm_6)
        return x


class AFNOArch(Arch):
    """Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    img_shape : Tuple[int, int]
        Input image dimensions (height, width)
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    patch_size : int, optional
        Size of image patchs, by default 16
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    num_blocks : int, optional
        Number of blocks in the frequency weight matrices, by default 4


    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size, H, W]`
    - Output variable tensor shape: :math:`[N, size, H, W]`

    Example
    -------
    >>> afno = .afno.AFNOArch([Key("x", size=2)], [Key("y", size=2)], (64, 64))
    >>> model = afno.make_node()
    >>> input = {"x": paddle.randn([20, 2, 64, 64])}
    >>> output = model.evaluate(input)
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        img_shape: Tuple[int, int],
        detach_keys: List[Key] = [],
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_blocks: int = 4,
    ) -> None:
        super().__init__(input_keys=input_keys, output_keys=output_keys)
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.detach_keys = detach_keys
        self.input_key_dict = {var.name: var.size for var in self.input_keys}
        self.output_key_dict = {var.name: var.size for var in self.output_keys}
        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values())
        self._impl = AFNONet(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=(patch_size, patch_size),
            img_size=img_shape,
            embed_dim=embed_dim,
            depth=depth,
            num_blocks=num_blocks,
        )

    def forward(self, in_vars: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        x = self.prepare_input(
            in_vars,
            mask=self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )
        y = self._impl(x)
        return self.prepare_output(
            y, output_var=self.output_key_dict, dim=1, output_scales=self.output_scales
        )
