import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
import numpy as np
from typing import Dict, List, Tuple
import itertools
import modulus.sym.models.fully_connected as fully_connected
import modulus.sym.models.layers as layers
from modulus.sym.models.interpolation import _grid_knn_idx, _hyper_cube_weighting, smooth_step_2, linear_step
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key
from modulus.sym.distributed import DistributedManager


class MultiresolutionHashNetArch(Arch):
    """Hash encoding network as seen in,

    Müller, Thomas, et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." arXiv preprint arXiv:2201.05989 (2022).
    A reference pytorch implementation can be found, https://github.com/yashbhalgat/HashNeRF-pytorch

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    activation_fn : layers.Activation = layers.Activation.SILU
        Activation function used by network.
    layer_size : int = 64
        Layer size for every hidden layer of the model.
    nr_layers : int = 3
        Number of hidden layers of the model.
    skip_connections : bool = False
        If true then apply skip connections every 2 hidden layers.
    weight_norm : bool = False
        Use weight norm on fully connected layers.
    adaptive_activations : bool = False
        If True then use an adaptive activation function as described here
        https://arxiv.org/abs/1906.01170.
    bounds : List[Tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0)]
        List of bounds for hash grid. Each element is a tuple
        of the upper and lower bounds.
    nr_levels :  int = 5
        Number of levels in the hash grid.
    nr_features_per_level : int = 2
        Number of features from each hash grid.
    log2_hashmap_size : int = 19
        Hash map size will be `2**log2_hashmap_size`.
    base_resolution : int = 2
        base resolution of hash grids.
    finest_resolution : int = 32
        Highest resolution of hash grids.
    """

    def __init__(self, input_keys: List[Key], output_keys: List[Key],
        detach_keys: List[Key]=[], activation_fn=layers.Activation.SILU,
        layer_size: int=64, nr_layers: int=3, skip_connections: bool=False,
        weight_norm: bool=True, adaptive_activations: bool=False, bounds:
        List[Tuple[float, float]]=[(-1.0, 1.0), (-1.0, 1.0)], nr_levels:
        int=16, nr_features_per_level: int=2, log2_hashmap_size: int=19,
        base_resolution: int=2, finest_resolution: int=32) ->None:
        super().__init__(input_keys=input_keys, output_keys=output_keys,
            detach_keys=detach_keys)
        self.xyzt_var = [x for x in self.input_key_dict if x in ['x', 'y',
            'z', 't']]
        self.params_var = [x for x in self.input_key_dict if x not in ['x',
            'y', 'z', 't']]
        in_features_xyzt = sum(v for k, v in self.input_key_dict.items() if
            k in self.xyzt_var)
        in_features_params = sum(v for k, v in self.input_key_dict.items() if
            k in self.params_var)
        in_features = in_features_xyzt + in_features_params
        out_features = sum(self.output_key_dict.values())
        if len(self.params_var) == 0:
            self.params_var = None
        self.place = DistributedManager().device
        self.bounds = bounds
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = paddle.to_tensor(data=[base_resolution],
            dtype='float32')
        self.finest_resolution = paddle.to_tensor(data=[finest_resolution],
            dtype='float32')
        self.nr_levels = nr_levels
        self.nr_features_per_level = nr_features_per_level
        self.embedding = paddle.nn.Embedding(num_embeddings=self.nr_levels *
            2 ** self.log2_hashmap_size, embedding_dim=self.
            nr_features_per_level)
        paddle.nn.initializer.Uniform(a=-0.001, b=0.001)(self.embedding.weight)
        self.b = np.exp((np.log(self.finest_resolution) - np.log(self.
            base_resolution)) / (nr_levels - 1))
        list_dx = []
        list_start = []
        list_resolution = []
        for i in range(self.nr_levels):
            resolution = int(np.floor(self.base_resolution * self.b ** i))
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            list_resolution.append(paddle.to_tensor(data=[resolution]).to(
                self.place).reshape([1, 1]))
            adjust_factor = (8253729 ** i + 2396403) % 32767 / 32767.0
            not_adjusted_dx = [((x[1] - x[0]) / (resolution - 1)) for x in
                self.bounds]
            grid = [(b[0] + (-2.0 + adjust_factor) * x, b[1] + (2.0 +
                adjust_factor) * x, resolution) for b, x in zip(self.bounds,
                not_adjusted_dx)]
            dx = paddle.to_tensor(data=[((x[1] - x[0]) / (x[2] - 1)) for x in
                grid]).to(self.place)
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            dx = dx.reshape([1, len(grid)])
            list_dx.append(dx)
            start = paddle.to_tensor(data=[val[0] for val in grid]).to(self
                .place)
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            start = start.reshape([1, len(grid)])
            list_start.append(start)
        self.resolutions = paddle.stack(x=list_resolution, axis=1)
        self.dx = paddle.stack(x=list_dx, axis=1)
        self.start = paddle.stack(x=list_start, axis=1)
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        self.hyper_cube = paddle.to_tensor(data=list(itertools.product(*(
            len(self.bounds) * [[0, 1]])))).to(self.place).reshape([1, 1, -1,
            len(bounds)])
        list_mul_factor = []
        mul_factor = paddle.to_tensor(data=[1], dtype='int32').to(self.place)
        for r in range(self.nr_levels):
            for d in range(len(self.bounds)):
                list_mul_factor.append(mul_factor.clone())
                mul_factor *= self.resolutions[0, r, 0]
                mul_factor %= 20731370
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        self.mul_factor = paddle.stack(x=list_mul_factor).reshape([1, self.
            nr_levels, 1, len(self.bounds)])
        self.fc = fully_connected.FullyConnectedArchCore(in_features=self.
            nr_features_per_level * nr_levels + in_features_params,
            layer_size=layer_size, out_features=out_features, nr_layers=
            nr_layers, skip_connections=skip_connections, activation_fn=
            activation_fn, adaptive_activations=adaptive_activations,
            weight_norm=weight_norm)

    def forward(self, in_vars: Dict[str, paddle.Tensor]) ->Dict[str, paddle.Tensor]:
        in_xyzt_var = self.prepare_input(in_vars, self.xyzt_var,
            detach_dict=self.detach_key_dict, dim=-1, input_scales=self.
            input_scales)
        unsqueezed_xyzt = paddle.unsqueeze(x=in_xyzt_var, axis=1)
        lower_indice = paddle.floor(x=(unsqueezed_xyzt - self.start) / self.dx
            ).astype(dtype='int32')
        all_indice = paddle.unsqueeze(x=lower_indice, axis=-2
            ) + self.hyper_cube
        lower_point = lower_indice * self.dx + self.start
        upper_point = lower_point + self.dx
        key = paddle.sum(x=all_indice * self.mul_factor, axis=-1)
        key = 10000003 * key + 124777 * paddle.bitwise_xor(x=key, y=paddle.
            to_tensor(data=3563504501))
        key = paddle.to_tensor(data=self.nr_levels * (1 << self.
            log2_hashmap_size) - 1).to(key.place) & key
        embed = self.embedding(key)
        smoothed_lower_point = smooth_step_2((unsqueezed_xyzt - lower_point
            ) / self.dx)
        smoother_upper_point = smooth_step_2(-(unsqueezed_xyzt -
            upper_point) / self.dx)
        weights = _hyper_cube_weighting(smoothed_lower_point,
            smoother_upper_point)
        hash_xyzt = paddle.sum(x=embed * weights, axis=-2)
        x = paddle.reshape(x=hash_xyzt, shape=[hash_xyzt.shape[0], -1])
        if self.params_var is not None:
            in_params_var = self.prepare_input(in_vars, self.params_var,
                detach_dict=self.detach_key_dict, dim=-1, input_scales=self
                .input_scales)
            x = paddle.concat(x=(x, in_params_var), axis=-1)
        x = self.fc(x)
        return self.prepare_output(x, self.output_key_dict, dim=-1,
            output_scales=self.output_scales)
