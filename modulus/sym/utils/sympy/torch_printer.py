import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
"""
Helper functions for converting sympy equations to pytorch
"""
from sympy import lambdify, Symbol, Derivative, Function, Basic, Add, Max, Min
from sympy.printing.str import StrPrinter
import numpy as np
import functools
from typing import List, Dict
from modulus.sym.constants import diff_str, tf_dt


def torch_lambdify(f, r, separable=False):
    """
    generates a PyTorch function from a sympy equation

    Parameters
    ----------
    f : Sympy Exp, float, int, bool
      the equation to convert to torch.
      If float, int, or bool this gets converted
      to a constant function of value `f`.
    r : list, dict
      A list of the arguments for `f`. If dict then
      the keys of the dict are used.

    Returns
    -------
    torch_f : PyTorch function
    """
    try:
        f = float(f)
    except:
        pass
    if isinstance(f, (float, int, bool)):

        def loop_lambda(constant):
            return lambda **x: paddle.zeros_like(x=next(iter(x.items()))[1]
                ) + constant
        lambdify_f = loop_lambda(f)
    else:
        vars = [k for k in r] if separable else [[k for k in r]]
        try:
            lambdify_f = lambdify(vars, f, [TORCH_SYMPY_PRINTER])
        except:
            lambdify_f = lambdify(vars, f, [TORCH_SYMPY_PRINTER])
    return lambdify_f


def _where_torch(conditions, x, y):
    if isinstance(x, (int, float)):
        x = float(x) * paddle.ones(shape=conditions.get_shape())
    if isinstance(y, (int, float)):
        y = float(y) * paddle.ones(shape=conditions.get_shape())
    return paddle.where(condition=conditions, x=x, y=y)


def _heaviside_torch(x):
    return paddle.maximum(x=paddle.sign(x=x), y=paddle.zeros(shape=[1]))


def _sqrt_torch(x):
    return paddle.sqrt(x=(x - 1e-06) * _heaviside_torch(x - 1e-06) + 1e-06)


def _or_torch(*x):
    return_value = x[0]
    for value in x:
        return_value = paddle.logical_or(x=return_value, y=value)
    return return_value


def _and_torch(*x):
    return_value = x[0]
    for value in x:
        return_value = paddle.logical_and(x=return_value, y=value)
    return return_value


# @torch.jit.script
def _min_jit(x: List[paddle.Tensor]):
    assert len(x) > 0
    min_tensor = x[0]
    for i in range(1, len(x)):
        min_tensor = paddle.minimum(x=min_tensor, y=x[i])
    return min_tensor


def _min_torch(*x):
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))
            device = value.place
    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = paddle.zeros(shape=tensor_shape) + value
        x_only_tensors.append(value)
    min_tensor, _ = paddle.minimum(x=paddle.stack(x=x_only_tensors, axis=-1
        ), y=-1)
    return min_tensor


# @torch.jit.script
def _max_jit(x: List[paddle.Tensor]):
    assert len(x) > 0
    max_tensor = x[0]
    for i in range(1, len(x)):
        max_tensor = paddle.maximum(x=max_tensor, y=x[i])
    return max_tensor


def _max_torch(*x):
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))
            device = value.place
    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = (paddle.zeros(shape=tensor_shape) + value).to(device)
        x_only_tensors.append(value)
    max_tensor, _ = paddle.maximum(x=paddle.stack(x=x_only_tensors, axis=-1
        ), y=-1)
    return max_tensor


def _dirac_delta_torch(x):
    return paddle.equal(x=x, y=0.0).to(tf_dt)


TORCH_SYMPY_PRINTER = {'abs': paddle.abs, 'Abs': paddle.abs, 'sign': paddle
    .sign, 'ceiling': paddle.ceil, 'floor': paddle.floor, 'log': paddle.log,
    'exp': paddle.exp, 'sqrt': _sqrt_torch, 'cos': paddle.cos, 'acos':
    paddle.acos, 'sin': paddle.sin, 'asin': paddle.asin, 'tan': paddle.tan,
    'atan': paddle.atan, 'atan2': paddle.atan2, 'cosh': paddle.cosh,
    'acosh': paddle.acosh, 'sinh': paddle.sinh, 'asinh': paddle.asinh,
    'tanh': paddle.tanh, 'atanh': paddle.atanh, 'erf': paddle.erf,
    'loggamma': paddle.lgamma, 'Min': _min_torch, 'Max': _max_torch,
    'Heaviside': _heaviside_torch, 'DiracDelta': _dirac_delta_torch,
    'logical_or': _or_torch, 'logical_and': _and_torch, 'where':
    _where_torch, 'pi': np.pi, 'conjugate': paddle.conj}


class CustomDerivativePrinter(StrPrinter):

    def _print_Function(self, expr):
        """
        Custom printing of the SymPy Derivative class.
        Instead of:
        D(x(t), t)
        We will print:
        x__t
        """
        return expr.func.__name__

    def _print_Derivative(self, expr):
        """
        Custom printing of the SymPy Derivative class.
        Instead of:
        D(x(t), t)
        We will print:
        x__t
        """
        prefix = str(expr.args[0].func)
        for expr in expr.args[1:]:
            prefix += expr[1] * (diff_str + str(expr[0]))
        return prefix


def _subs_derivatives(expr):
    while True:
        try:
            deriv = expr.atoms(Derivative).pop()
            new_fn_name = str(deriv)
            expr = expr.subs(deriv, Function(new_fn_name)(*deriv.free_symbols))
        except:
            break
    while True:
        try:
            fn = {fn for fn in expr.atoms(Function) if fn.class_key()[1] == 0
                }.pop()
            new_symbol_name = str(fn)
            expr = expr.subs(fn, Symbol(new_symbol_name))
        except:
            break
    return expr


Basic.__str__ = lambda self: CustomDerivativePrinter().doprint(self)


class SympyToTorch(paddle.nn.Layer):

    def __init__(self, sympy_expr, name: str, freeze_terms: List[int]=[],
        detach_names: List[str]=[]):
        super().__init__()
        self.keys = sorted([k.name for k in sympy_expr.free_symbols])
        self.freeze_terms = freeze_terms
        if not self.freeze_terms:
            self.torch_expr = torch_lambdify(sympy_expr, self.keys)
        else:
            assert all(x < len(Add.make_args(sympy_expr)) for x in freeze_terms
                ), 'The freeze term index cannot be larger than the total terms in the expression'
            self.torch_expr = []
            for i in range(len(Add.make_args(sympy_expr))):
                self.torch_expr.append(torch_lambdify(Add.make_args(
                    sympy_expr)[i], self.keys))
            self.freeze_list = list(self.torch_expr[i] for i in freeze_terms)
        self.name = name
        self.detach_names = detach_names

    def forward(self, var: Dict[str, paddle.Tensor]) ->Dict[str, paddle.Tensor
        ]:
        args = [(var[k].detach() if k in self.detach_names else var[k]) for
            k in self.keys]
        if not self.freeze_terms:
            output = self.torch_expr(args)
        else:
            output = paddle.zeros_like(x=var[self.keys[0]])
            for i, expr in enumerate(self.torch_expr):
                if expr in self.freeze_list:
                    output += expr(args).detach()
                else:
                    output += expr(args)
        return {self.name: output}
