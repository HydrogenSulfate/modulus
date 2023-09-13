import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
from typing import List
from typing import Tuple
import numpy as np


class SpectralConv1d(paddle.nn.Layer):

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        out_66 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, 2]).shape, dtype=paddle
            .empty(shape=[in_channels, out_channels, self.modes1, 2]).numpy
            ().dtype, default_initializer=paddle.nn.initializer.Assign(
            paddle.empty(shape=[in_channels, out_channels, self.modes1, 2])))
        out_66.stop_gradient = not True
        self.weights1 = out_66
        self.reset_parameters()

    def compl_mul1d(self, input: paddle.Tensor, weights: paddle.Tensor
        ) ->paddle.Tensor:
        cweights = paddle.as_complex(x=weights)
        return paddle.einsum('bix,iox->box', input, cweights)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        bsize = x.shape[0]
        x_ft = paddle.fft.rfft(x=x)
        out_ft = paddle.zeros(shape=[bsize, self.out_channels, x.shape[-1] //
            2 + 1], dtype='complex64')
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.
            modes1], self.weights1)
        x = paddle.fft.irfft(x=out_ft, n=x.shape[-1])
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * paddle.rand(shape=self.weights1.
            data.shape)


class SpectralConv2d(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        out_67 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, self.modes2, 2]).shape,
            dtype=paddle.empty(shape=[in_channels, out_channels, self.
            modes1, self.modes2, 2]).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.empty(shape=[in_channels,
            out_channels, self.modes1, self.modes2, 2])))
        out_67.stop_gradient = not True
        self.weights1 = out_67
        out_68 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, self.modes2, 2]).shape,
            dtype=paddle.empty(shape=[in_channels, out_channels, self.
            modes1, self.modes2, 2]).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.empty(shape=[in_channels,
            out_channels, self.modes1, self.modes2, 2])))
        out_68.stop_gradient = not True
        self.weights2 = out_68
        self.reset_parameters()

    def compl_mul2d(self, input: paddle.Tensor, weights: paddle.Tensor
        ) ->paddle.Tensor:
        cweights = paddle.as_complex(x=weights)
        return paddle.einsum('bixy,ioxy->boxy', input, cweights)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfft2(x=x)
        out_ft = paddle.zeros(shape=[batchsize, self.out_channels, x.shape[
            -2], x.shape[-1] // 2 + 1], dtype='complex64')
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:,
            :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:,
            :, -self.modes1:, :self.modes2], self.weights2)
        x = paddle.fft.irfft2(x=out_ft, s=(x.shape[-2], x.shape[-1]))
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * paddle.rand(shape=self.weights1.
            data.shape)
        self.weights2.data = self.scale * paddle.rand(shape=self.weights2.
            data.shape)


class SpectralConv3d(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        out_69 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, self.modes2, self.
            modes3, 2]).shape, dtype=paddle.empty(shape=[in_channels,
            out_channels, self.modes1, self.modes2, self.modes3, 2]).numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(
            paddle.empty(shape=[in_channels, out_channels, self.modes1,
            self.modes2, self.modes3, 2])))
        out_69.stop_gradient = not True
        self.weights1 = out_69
        out_70 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, self.modes2, self.
            modes3, 2]).shape, dtype=paddle.empty(shape=[in_channels,
            out_channels, self.modes1, self.modes2, self.modes3, 2]).numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(
            paddle.empty(shape=[in_channels, out_channels, self.modes1,
            self.modes2, self.modes3, 2])))
        out_70.stop_gradient = not True
        self.weights2 = out_70
        out_71 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, self.modes2, self.
            modes3, 2]).shape, dtype=paddle.empty(shape=[in_channels,
            out_channels, self.modes1, self.modes2, self.modes3, 2]).numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(
            paddle.empty(shape=[in_channels, out_channels, self.modes1,
            self.modes2, self.modes3, 2])))
        out_71.stop_gradient = not True
        self.weights3 = out_71
        out_72 = paddle.create_parameter(shape=paddle.empty(shape=[
            in_channels, out_channels, self.modes1, self.modes2, self.
            modes3, 2]).shape, dtype=paddle.empty(shape=[in_channels,
            out_channels, self.modes1, self.modes2, self.modes3, 2]).numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(
            paddle.empty(shape=[in_channels, out_channels, self.modes1,
            self.modes2, self.modes3, 2])))
        out_72.stop_gradient = not True
        self.weights4 = out_72
        self.reset_parameters()

    def compl_mul3d(self, input: paddle.Tensor, weights: paddle.Tensor
        ) ->paddle.Tensor:
        cweights = paddle.as_complex(x=weights)
        return paddle.einsum('bixyz,ioxyz->boxyz', input, cweights)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfftn(x=x, axes=[-3, -2, -1])
        out_ft = paddle.zeros(shape=[batchsize, self.out_channels, x.shape[
            -3], x.shape[-2], x.shape[-1] // 2 + 1], dtype='complex64')
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3
            ] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :
            self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3
            ] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :
            self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3
            ] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :
            self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3
            ] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :
            self.modes3], self.weights4)
        x = paddle.fft.irfftn(x=out_ft, s=(x.shape[-3], x.shape[-2], x.
            shape[-1]))
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * paddle.rand(shape=self.weights1.
            data.shape)
        self.weights2.data = self.scale * paddle.rand(shape=self.weights2.
            data.shape)
        self.weights3.data = self.scale * paddle.rand(shape=self.weights3.
            data.shape)
        self.weights4.data = self.scale * paddle.rand(shape=self.weights4.
            data.shape)


def fourier_derivatives(x: paddle.Tensor, l: List[float]) ->Tuple[paddle.Tensor,
    paddle.Tensor]:
    assert len(x.shape) - 2 == len(l), "input shape doesn't match domain dims"
    pi = float(np.pi)
    batchsize = x.shape[0]
    n = x.shape[2:]
    dim = len(l)
    device = x.place
    x_h = paddle.fft.fftn(x=x, axes=list(range(2, dim + 2)))
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(paddle.concat(x=(paddle.arange(start=0, end=nx // 2,
            step=1), paddle.arange(start=-nx // 2, end=0, step=1)), axis=0)
            .reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1]))
    j = paddle.complex(real=paddle.to_tensor(data=[0.0], place=device),
        imag=paddle.to_tensor(data=[1.0], place=device))
    wx_h = [(j * k_x_i * x_h * (2 * pi / l[i])) for i, k_x_i in enumerate(k_x)]
    wxx_h = [(j * k_x_i * wx_h_i * (2 * pi / l[i])) for i, (wx_h_i, k_x_i) in
        enumerate(zip(wx_h, k_x))]
    wx = paddle.concat(x=[paddle.fft.ifftn(x=wx_h_i, axes=list(range(2, dim +
        2))).real() for wx_h_i in wx_h], axis=1)
    wxx = paddle.concat(x=[paddle.fft.ifftn(x=wxx_h_i, axes=list(range(2, 
        dim + 2))).real() for wxx_h_i in wxx_h], axis=1)
    return wx, wxx


# @torch.jit.ignore
def calc_latent_derivatives(x: paddle.Tensor, domain_length: List[int]=2
    ) ->Tuple[List[paddle.Tensor], List[paddle.Tensor]]:
    dim = len(x.shape) - 2
    padd = [((i - 1) // 2) for i in list(x.shape[2:])]
    domain_length = [(domain_length[i] * (2 * padd[i] + x.shape[i + 2]) / x
        .shape[i + 2]) for i in range(dim)]
    padding = padd + padd
    x_p = paddle.nn.functional.pad(x, padding, mode='replicate')
    dx, ddx = fourier_derivatives(x_p, domain_length)
    if len(x.shape) == 3:
        dx = dx[..., padd[0]:-padd[0]]
        ddx = ddx[..., padd[0]:-padd[0]]
        dx_list = paddle.split(x=dx, num_or_sections=x.shape[1], axis=1)
        ddx_list = paddle.split(x=ddx, num_or_sections=x.shape[1], axis=1)
    elif len(x.shape) == 4:
        dx = dx[..., padd[0]:-padd[0], padd[1]:-padd[1]]
        ddx = ddx[..., padd[0]:-padd[0], padd[1]:-padd[1]]
        dx_list = paddle.split(x=dx, num_or_sections=x.shape[1], axis=1)
        ddx_list = paddle.split(x=ddx, num_or_sections=x.shape[1], axis=1)
    else:
        dx = dx[..., padd[0]:-padd[0], padd[1]:-padd[1], padd[2]:-padd[2]]
        ddx = ddx[..., padd[0]:-padd[0], padd[1]:-padd[1], padd[2]:-padd[2]]
        dx_list = paddle.split(x=dx, num_or_sections=x.shape[1], axis=1)
        ddx_list = paddle.split(x=ddx, num_or_sections=x.shape[1], axis=1)
    return dx_list, ddx_list


def first_order_pino_grads(u: paddle.Tensor, ux: List[paddle.Tensor], weights_1:
    paddle.Tensor, weights_2: paddle.Tensor, bias_1: paddle.Tensor) ->Tuple[
    paddle.Tensor]:
    dim = len(u.shape) - 2
    dim_str = 'xyz'[:dim]
    if dim == 1:
        u_hidden = paddle.nn.functional.conv1d(x=u, weight=weights_1, bias=
            bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1)
        u_hidden = paddle.nn.functional.conv2d(x=u, weight=weights_1, bias=
            bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(axis=-1).unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1).unsqueeze(axis=-1)
        u_hidden = paddle.nn.functional.conv3d(x=u, weight=weights_1, bias=
            bias_1)
    diff_tanh = 1 / paddle.cosh(x=u_hidden) ** 2
    diff_fg = paddle.einsum('mi' + dim_str + ',bm' + dim_str + ',km' +
        dim_str + '->bi' + dim_str, weights_1, diff_tanh, weights_2)
    vx = [paddle.einsum('bi' + dim_str + ',bi' + dim_str + '->b' + dim_str,
        diff_fg, w) for w in ux]
    vx = [paddle.unsqueeze(x=w, axis=1) for w in vx]
    return vx


def second_order_pino_grads(u: paddle.Tensor, ux: paddle.Tensor, uxx:
    paddle.Tensor, weights_1: paddle.Tensor, weights_2: paddle.Tensor,
    bias_1: paddle.Tensor) ->Tuple[paddle.Tensor]:
    dim = len(u.shape) - 2
    dim_str = 'xyz'[:dim]
    if dim == 1:
        u_hidden = paddle.nn.functional.conv1d(x=u, weight=weights_1, bias=
            bias_1)
    elif dim == 2:
        weights_1 = weights_1.unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1)
        u_hidden = paddle.nn.functional.conv2d(x=u, weight=weights_1, bias=
            bias_1)
    elif dim == 3:
        weights_1 = weights_1.unsqueeze(axis=-1).unsqueeze(axis=-1)
        weights_2 = weights_2.unsqueeze(axis=-1).unsqueeze(axis=-1)
        u_hidden = paddle.nn.functional.conv3d(x=u, weight=weights_1, bias=
            bias_1)
    diff_tanh = 1 / paddle.cosh(x=u_hidden) ** 2
    diff_fg = paddle.einsum('mi' + dim_str + ',bm' + dim_str + ',km' +
        dim_str + '->bi' + dim_str, weights_1, diff_tanh, weights_2)
    diff_diff_tanh = -2 * diff_tanh * paddle.tanh(x=u_hidden)
    vxx1 = [paddle.einsum('bi' + dim_str + ',mi' + dim_str + ',bm' +
        dim_str + ',mj' + dim_str + ',bj' + dim_str + '->b' + dim_str, w,
        weights_1, weights_2 * diff_diff_tanh, weights_1, w) for w in ux]
    vxx2 = [paddle.einsum('bi' + dim_str + ',bi' + dim_str + '->b' +
        dim_str, diff_fg, w) for w in uxx]
    vxx = [paddle.unsqueeze(x=a + b, axis=1) for a, b in zip(vxx1, vxx2)]
    return vxx
