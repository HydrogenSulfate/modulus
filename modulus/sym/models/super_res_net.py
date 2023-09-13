import paddle
""""""
"""
SRResNet model. This code was modified from, https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

The following license is provided from their source,

MIT License

Copyright (c) 2020 Sagar Vinodababu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
from typing import List, Dict
from modulus.sym.key import Key
from modulus.sym.models.arch import Arch
from modulus.sym.models.layers import Activation, get_activation_fn
Tensor = paddle.Tensor


class ConvolutionalBlock3d(paddle.nn.Layer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size:
        int, stride: int=1, batch_norm: bool=False, activation_fn:
        Activation=Activation.IDENTITY):
        super().__init__()
        activation_fn = get_activation_fn(activation_fn)
        layers = list()
        layers.append(paddle.nn.Conv3D(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=kernel_size // 2))
        if batch_norm is True:
            layers.append(paddle.nn.BatchNorm3D(num_features=out_channels))
        self.activation_fn = get_activation_fn(activation_fn)
        self.conv_block = paddle.nn.Sequential(*layers)

    def forward(self, input: Tensor) ->Tensor:
        output = self.activation_fn(self.conv_block(input))
        return output


class PixelShuffle3d(paddle.nn.Layer):

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, input: Tensor) ->Tensor:
        batch_size, channels, in_depth, in_height, in_width = input.shape
        nOut = int(channels // self.scale ** 3)
        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        input_view = input.reshape([batch_size, nOut, self.scale, self.scale,
            self.scale, in_depth, in_height, in_width])
        output = input_view.transpose(perm=[0, 1, 5, 2, 6, 3, 7, 4])
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        return output.reshape([batch_size, nOut, out_depth, out_height, out_width])


class SubPixelConvolutionalBlock3d(paddle.nn.Layer):

    def __init__(self, kernel_size: int=3, conv_layer_size: int=64,
        scaling_factor: int=2):
        super().__init__()
        self.conv = paddle.nn.Conv3D(in_channels=conv_layer_size,
            out_channels=conv_layer_size * scaling_factor ** 3, kernel_size
            =kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = PixelShuffle3d(scaling_factor)
        self.prelu = paddle.nn.PReLU()

    def forward(self, input: Tensor) ->Tensor:
        output = self.conv(input)
        output = self.pixel_shuffle(output)
        output = self.prelu(output)
        return output


class ResidualConvBlock3d(paddle.nn.Layer):

    def __init__(self, n_layers: int=1, kernel_size: int=3, conv_layer_size:
        int=64, activation_fn: Activation=Activation.IDENTITY):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers.append(ConvolutionalBlock3d(in_channels=conv_layer_size,
                out_channels=conv_layer_size, kernel_size=kernel_size,
                batch_norm=True, activation_fn=activation_fn))
        layers.append(ConvolutionalBlock3d(in_channels=conv_layer_size,
            out_channels=conv_layer_size, kernel_size=kernel_size,
            batch_norm=True))
        self.conv_layers = paddle.nn.Sequential(*layers)

    def forward(self, input: Tensor) ->Tensor:
        residual = input
        output = self.conv_layers(input)
        output = output + residual
        return output


class SRResNetArch(Arch):
    """3D super resolution network

    Based on the implementation:  https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    large_kernel_size : int, optional
        convolutional kernel size for first and last convolution, by default 7
    small_kernel_size : int, optional
        convolutional kernel size for internal convolutions, by default 3
    conv_layer_size : int, optional
        Latent channel size, by default 32
    n_resid_blocks : int, optional
        Number of residual blocks before , by default 8
    scaling_factor : int, optional
        Scaling factor to increase the output feature size compared to the input (2, 4, or 8), by default 8
    activation_fn : Activation, optional
        Activation function, by default Activation.PRELU
    """

    def __init__(self, input_keys: List[Key], output_keys: List[Key],
        detach_keys: List[Key]=[], large_kernel_size: int=7,
        small_kernel_size: int=3, conv_layer_size: int=32, n_resid_blocks:
        int=8, scaling_factor: int=8, activation_fn: Activation=Activation.
        PRELU):
        super().__init__(input_keys=input_keys, output_keys=output_keys,
            detach_keys=detach_keys)
        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values())
        self.var_dim = 1
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8
            }, 'The scaling factor must be 2, 4, or 8!'
        self.conv_block1 = ConvolutionalBlock3d(in_channels=in_channels,
            out_channels=conv_layer_size, kernel_size=large_kernel_size,
            batch_norm=False, activation_fn=activation_fn)
        self.residual_blocks = paddle.nn.Sequential(*[ResidualConvBlock3d(
            n_layers=2, kernel_size=small_kernel_size, conv_layer_size=
            conv_layer_size, activation_fn=activation_fn) for i in range(
            n_resid_blocks)])
        self.conv_block2 = ConvolutionalBlock3d(in_channels=conv_layer_size,
            out_channels=conv_layer_size, kernel_size=small_kernel_size,
            batch_norm=True)
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = paddle.nn.Sequential(*[
            SubPixelConvolutionalBlock3d(kernel_size=small_kernel_size,
            conv_layer_size=conv_layer_size, scaling_factor=2) for i in
            range(n_subpixel_convolution_blocks)])
        self.conv_block3 = ConvolutionalBlock3d(in_channels=conv_layer_size,
            out_channels=out_channels, kernel_size=large_kernel_size,
            batch_norm=False)

    def forward(self, in_vars: Dict[str, Tensor]) ->Dict[str, Tensor]:
        input = self.prepare_input(in_vars, self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict, dim=1, input_scales=self.
            input_scales, periodicity=self.periodicity)
        output = self.conv_block1(input)
        residual = output
        output = self.residual_blocks(output)
        output = self.conv_block2(output)
        output = output + residual
        output = self.subpixel_convolutional_blocks(output)
        output = self.conv_block3(output)
        return self.prepare_output(output, self.output_key_dict, dim=1,
            output_scales=self.output_scales)
