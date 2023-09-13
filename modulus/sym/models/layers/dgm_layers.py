import paddle
from typing import Callable
from typing import Optional
from typing import Union
from .weight_norm import WeightNormLinear
from .activation import Activation, get_activation_fn


class DGMLayer(paddle.nn.Layer):

    def __init__(self, in_features_1: int, in_features_2: int, out_features:
        int, activation_fn: Union[Activation, Callable[[paddle.Tensor], paddle.Tensor]]=
        Activation.IDENTITY, weight_norm: bool=False, activation_par:
        Optional[paddle.Tensor]=None) ->None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.weight_norm = weight_norm
        self.activation_par = activation_par
        if weight_norm:
            self.linear_1 = WeightNormLinear(in_features_1, out_features,
                bias=False)
            self.linear_2 = WeightNormLinear(in_features_2, out_features,
                bias=False)
        else:
            self.linear_1 = paddle.nn.Linear(in_features=in_features_1,
                out_features=out_features, bias_attr=False)
            self.linear_2 = paddle.nn.Linear(in_features=in_features_2,
                out_features=out_features, bias_attr=False)
        out_59 = paddle.create_parameter(shape=paddle.empty(shape=
            out_features).shape, dtype=paddle.empty(shape=out_features).
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (paddle.empty(shape=out_features)))
        out_59.stop_gradient = not True
        self.bias = out_59
        self.reset_parameters()

    def exec_activation_fn(self, x: paddle.Tensor) ->paddle.Tensor:
        return self.callable_activation_fn(x)

    def reset_parameters(self) ->None:
        paddle.nn.initializer.XavierNormal()(self.linear_1.weight)
        paddle.nn.initializer.XavierNormal()(self.linear_2.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.bias)
        if self.weight_norm:
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(self.linear_1.weight_g)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(self.linear_2.weight_g)

    def forward(self, input_1: paddle.Tensor, input_2: paddle.Tensor
        ) ->paddle.Tensor:
        x = self.linear_1(input_1) + self.linear_2(input_2) + self.bias
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x
