import paddle
import logging
from typing import Callable
from typing import Optional
from typing import Union
from .weight_norm import WeightNormLinear
from .activation import Activation, get_activation_fn
logger = logging.getLogger(__name__)


class FCLayer(paddle.nn.Layer):

    def __init__(self, in_features: int, out_features: int, activation_fn:
        Union[Activation, Callable[[paddle.Tensor], paddle.Tensor]]=Activation.IDENTITY,
        weight_norm: bool=False, activation_par: Optional[paddle.
        Tensor]=None) ->None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn,
            out_features=out_features)
        self.weight_norm = weight_norm
        self.activation_par = activation_par
        if weight_norm:
            self.linear = WeightNormLinear(in_features, out_features, bias=True
                )
        else:
            self.linear = paddle.nn.Linear(in_features=in_features,
                out_features=out_features, bias_attr=True)
        self.reset_parameters()

    def exec_activation_fn(self, x: paddle.Tensor) ->paddle.Tensor:
        return self.callable_activation_fn(x)

    def reset_parameters(self) ->None:
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.linear.bias)
        paddle.nn.initializer.XavierUniform()(self.linear.weight)
        if self.weight_norm:
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(self.linear.weight_g)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.linear(x)
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


class ConvFCLayer(paddle.nn.Layer):

    def __init__(self, activation_fn: Union[Activation, Callable[[paddle.Tensor],
        paddle.Tensor]]=Activation.IDENTITY, activation_par: Optional[paddle.Tensor]=None) ->None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.activation_par = activation_par

    def exec_activation_fn(self, x: paddle.Tensor) ->paddle.Tensor:
        return self.callable_activation_fn(x)

    def apply_activation(self, x: paddle.Tensor) ->paddle.Tensor:
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):

    def __init__(self, in_features: int, out_features: int, activation_fn:
        Union[Activation, Callable[[paddle.Tensor], paddle.Tensor]]=Activation.IDENTITY,
        weight_norm: bool=False, activation_par: Optional[paddle.Tensor]=None) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = paddle.nn.Conv1D(in_channels=in_features, out_channels=
            out_features, kernel_size=1, bias_attr=True)
        self.reset_parameters()
        if weight_norm:
            logger.warn('Weight norm not supported for Conv FC layers')

    def reset_parameters(self) ->None:
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.conv.bias)
        paddle.nn.initializer.XavierUniform()(self.conv.weight)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):

    def __init__(self, in_channels: int, out_channels: int, activation_fn:
        Union[Activation, Callable[[paddle.Tensor], paddle.Tensor]]=Activation.IDENTITY,
        activation_par: Optional[paddle.Tensor]=None) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = paddle.nn.Conv2D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, bias_attr=True)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.conv.bias)
        self.conv.bias.stop_gradient = not False
        paddle.nn.initializer.XavierUniform()(self.conv.weight)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):

    def __init__(self, in_channels: int, out_channels: int, activation_fn:
        Union[Activation, Callable[[paddle.Tensor], paddle.Tensor]]=Activation.IDENTITY,
        activation_par: Optional[paddle.Tensor]=None) ->None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = paddle.nn.Conv3D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, bias_attr=True)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.conv.bias)
        paddle.nn.initializer.XavierUniform()(self.conv.weight)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x
