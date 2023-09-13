"""Basic equations
"""
from sympy import Symbol, Function, Number
from modulus.sym.eq.pde import PDE
from modulus.sym.node import Node


class NormalDotVec(PDE):
    """
    Normal dot velocity

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """
    name = 'NormalDotVec'

    def __init__(self, vec=['u', 'v', 'w']):
        normal = [Symbol('normal_x'), Symbol('normal_y'), Symbol('normal_z')]
        self.equations = {}
        self.equations['normal_dot_vel'] = 0
        for v, n in zip(vec, normal):
            self.equations['normal_dot_vel'] += Symbol(v) * n


class GradNormal(PDE):
    """
    Implementation of the gradient boundary condition

    Parameters
    ==========
    T : str
        The dependent variable.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Examples
    ========
    >>> gn = ns = GradNormal(T='T')
    >>> gn.pprint()
      normal_gradient_T: normal_x*T__x + normal_y*T__y + normal_z*T__z
    """
    name = 'GradNormal'

    def __init__(self, T, dim=3, time=True):
        self.T = T
        self.dim = dim
        self.time = time
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        normal_x = Symbol('normal_x')
        normal_y = Symbol('normal_y')
        normal_z = Symbol('normal_z')
        t = Symbol('t')
        input_variables = {'x': x, 'y': y, 'z': z, 't': t}
        if self.dim == 1:
            input_variables.pop('y')
            input_variables.pop('z')
        elif self.dim == 2:
            input_variables.pop('z')
        if not self.time:
            input_variables.pop('t')
        T = Function(T)(*input_variables)
        self.equations = {}
        """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        self.equations['normal_gradient_' + self.T] = normal_x * T.diff(x
            ) + normal_y * T.diff(y) + normal_z * T.diff(z)


class Curl(PDE):
    """
    del cross vector operator

    Parameters
    ==========
    vector : tuple of 3 Sympy Exprs, floats, ints or strings
        This will be the vector to take the curl of.
    curl_name : tuple of 3 strings
        These will be the output names of the curl operations.

    Examples
    ========
    >>> c = Curl((0,0,'phi'), ('u','v','w'))
    >>> c.pprint()
      u: phi__y
      v: -phi__x
      w: 0
    """
    name = 'Curl'

    def __init__(self, vector, curl_name=['u', 'v', 'w']):
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        input_variables = {'x': x, 'y': y, 'z': z}
        v_0 = vector[0]
        v_1 = vector[1]
        v_2 = vector[2]
        if type(v_0) is str:
            v_0 = Function(v_0)(*input_variables)
        elif type(v_0) in [float, int]:
            v_0 = Number(v_0)
        if type(v_1) is str:
            v_1 = Function(v_1)(*input_variables)
        elif type(v_1) in [float, int]:
            v_1 = Number(v_1)
        if type(v_2) is str:
            v_2 = Function(v_2)(*input_variables)
        elif type(v_2) in [float, int]:
            v_2 = Number(v_2)
        """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        curl_0 = v_2.diff(y) - v_1.diff(z)
        """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        curl_1 = v_0.diff(z) - v_2.diff(x)
        """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        curl_2 = v_1.diff(x) - v_0.diff(y)
        self.equations = {}
        self.equations[curl_name[0]] = curl_0
        self.equations[curl_name[1]] = curl_1
        self.equations[curl_name[2]] = curl_2
