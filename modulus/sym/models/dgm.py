import paddle
from typing import List, Dict
import modulus.sym.models.layers as layers
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key


class DGMArch(Arch):
    """
    A variation of the fully connected network.
    Reference: Sirignano, J. and Spiliopoulos, K., 2018.
    DGM: A deep learning algorithm for solving partial differential equations.
    Journal of computational physics, 375, pp.1339-1364.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 512
        Layer size for every hidden layer of the model.
    nr_layers : int = 6
        Number of hidden layers of the model.
    skip_connections : bool = False
        If true then apply skip connections every 2 hidden layers.
    activation_fn : layers.Activation = layers.Activation.SILU
        Activation function used by network.
    adaptive_activations : bool = False
        If True then use an adaptive activation function as described here
        https://arxiv.org/abs/1906.01170.
    weight_norm : bool = True
        Use weight norm on fully connected layers.
    """

    def __init__(self, input_keys: List[Key], output_keys: List[Key],
        detach_keys: List[Key]=[], layer_size: int=512, nr_layers: int=6,
        activation_fn=layers.Activation.SIN, adaptive_activations: bool=
        False, weight_norm: bool=True) ->None:
        super().__init__(input_keys=input_keys, output_keys=output_keys,
            detach_keys=detach_keys)
        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())
        if adaptive_activations:
            out_54 = paddle.create_parameter(shape=paddle.ones(shape=[1]).
                shape, dtype=paddle.ones(shape=[1]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.
                ones(shape=[1])))
            out_54.stop_gradient = not True
            activation_par = out_54
        else:
            activation_par = None
        self.fc_start = layers.FCLayer(in_features=in_features,
            out_features=layer_size, activation_fn=activation_fn,
            weight_norm=weight_norm)
        self.dgm_layers = paddle.nn.LayerList()
        for _ in range(nr_layers - 1):
            single_layer = {}
            for key in ['z', 'g', 'r', 'h']:
                single_layer[key] = layers.DGMLayer(in_features_1=
                    in_features, in_features_2=layer_size, out_features=
                    layer_size, activation_fn=activation_fn, weight_norm=
                    weight_norm, activation_par=activation_par)
            self.dgm_layers.append(paddle.nn.LayerDict(sublayers=single_layer))
        self.fc_end = layers.FCLayer(in_features=layer_size, out_features=
            out_features, activation_fn=layers.Activation.IDENTITY,
            weight_norm=False, activation_par=None)

    def _tensor_forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.process_input(x, self.input_scales_tensor, periodicity=
            self.periodicity, input_dict=self.input_key_dict, dim=-1)
        s = self.fc_start(x)
        for layer in self.dgm_layers:
            z = layer['z'](x, s)
            g = layer['g'](x, s)
            r = layer['r'](x, s)
            h = layer['h'](x, s * r)
            s = h - g * h + z * s
        x = self.fc_end(s)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) ->Dict[str, Tensor]:
        x = self.concat_input(in_vars, self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict, dim=-1)
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) ->Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(in_vars, self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict, dim=-1, input_scales=self.
            input_scales)
        s = self.fc_start(x)
        for layer in self.dgm_layers:
            z = layer['z'](x, s)
            g = layer['g'](x, s)
            r = layer['r'](x, s)
            h = layer['h'](x, s * r)
            s = h - g * h + z * s
        x = self.fc_end(s)
        return self.prepare_output(x, self.output_key_dict, dim=-1,
            output_scales=self.output_scales)
