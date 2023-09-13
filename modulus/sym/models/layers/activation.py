import paddle
import enum
from typing import Callable
from typing import Union
from typing import List
from modulus.sym.manager import JitManager, JitArchMode


class ActivationMeta(enum.EnumMeta):

    def __getitem__(self, name):
        try:
            return super().__getitem__(name.upper())
        except KeyError as error:
            raise KeyError(f'Invalid activation function {name}')


class Activation(enum.Enum, metaclass=ActivationMeta):
    ELU = enum.auto()
    LEAKY_RELU = enum.auto()
    MISH = enum.auto()
    RELU = enum.auto()
    GELU = enum.auto()
    SELU = enum.auto()
    PRELU = enum.auto()
    SIGMOID = enum.auto()
    SILU = enum.auto()
    SIN = enum.auto()
    SQUAREPLUS = enum.auto()
    SOFTPLUS = enum.auto()
    TANH = enum.auto()
    STAN = enum.auto()
    IDENTITY = enum.auto()


def identity(x: paddle.Tensor) ->paddle.Tensor:
    return x


def squareplus(x: paddle.Tensor) ->paddle.Tensor:
    b = 4
    return 0.5 * (x + paddle.sqrt(x=x * x + b))


def gelu(x: paddle.Tensor) ->paddle.Tensor:
    return 0.5 * x * (1.0 + paddle.tanh(x=x * 0.7978845608 * (1.0 + 
        0.044715 * x * x)))


class Stan(paddle.nn.Layer):
    """
    Self-scalable Tanh (Stan)
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.
    """

    def __init__(self, out_features=1):
        super().__init__()
        out_58 = paddle.create_parameter(shape=paddle.ones(shape=
            out_features).shape, dtype=paddle.ones(shape=out_features).
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (paddle.ones(shape=out_features)))
        out_58.stop_gradient = not True
        self.beta = out_58

    def forward(self, x):
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f'The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}'
                )
        return paddle.tanh(x=x) * (1.0 + self.beta * x)


def get_activation_fn(activation: Union[Activation, Callable[[paddle.Tensor],
    paddle.Tensor]], module: bool=False, **kwargs) ->Callable[[paddle.Tensor], paddle.Tensor]:
    activation_mapping = {Activation.ELU: paddle.nn.functional.elu,
        Activation.LEAKY_RELU: paddle.nn.functional.leaky_relu, Activation.
        MISH: paddle.nn.functional.mish, Activation.RELU: paddle.nn.
        functional.relu, Activation.GELU: paddle.nn.functional.gelu,
        Activation.SELU: paddle.nn.functional.selu, Activation.SIGMOID:
        paddle.nn.functional.sigmoid, Activation.SILU: paddle.nn.functional
        .silu, Activation.SIN: paddle.sin, Activation.SQUAREPLUS:
        squareplus, Activation.SOFTPLUS: paddle.nn.functional.softplus,
        Activation.TANH: paddle.tanh, Activation.IDENTITY: identity}
    module_activation_mapping = {Activation.ELU: paddle.nn.ELU, Activation.
        LEAKY_RELU: paddle.nn.LeakyReLU, Activation.MISH: paddle.nn.Mish,
        Activation.RELU: paddle.nn.ReLU, Activation.GELU: paddle.nn.GELU,
        Activation.SELU: paddle.nn.SELU, Activation.PRELU: paddle.nn.PReLU,
        Activation.SIGMOID: paddle.nn.Sigmoid, Activation.SILU: paddle.nn.
        Silu, Activation.TANH: paddle.nn.Tanh, Activation.STAN: Stan}
    if activation in activation_mapping and not module:
        activation_fn_ = activation_mapping[activation]

        def activation_fn(x: paddle.Tensor) ->paddle.Tensor:
            return activation_fn_(x)
    elif activation in module_activation_mapping:
        activation_fn = module_activation_mapping[activation](**kwargs)
    else:
        activation_fn = activation
    if JitManager().enabled and JitManager(
        ).arch_mode == JitArchMode.ONLY_ACTIVATION:
        activation_fn = paddle.jit.script(activation_fn)
    return activation_fn
