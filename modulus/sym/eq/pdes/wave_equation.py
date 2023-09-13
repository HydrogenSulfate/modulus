"""Wave equation
Reference: https://en.wikipedia.org/wiki/Wave_equation
"""
from sympy import Symbol, Function, Number
from modulus.sym.eq.pde import PDE


class WaveEquation(PDE):
    """
    Wave equation

    Parameters
    ==========
    u : str
        The dependent variable.
    c : float, Sympy Symbol/Expr, str
        Wave speed coefficient. If `c` is a str then it is
        converted to Sympy Function of form 'c(x,y,z,t)'.
        If 'c' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the wave equation.

    Examples
    ========
    >>> we = WaveEquation(c=0.8, dim=3)
    >>> we.pprint()
      wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    >>> we = WaveEquation(c='c', dim=2, time=False)
    >>> we.pprint()
      wave_equation: -c**2*u__x__x - c**2*u__y__y - 2*c*c__x*u__x - 2*c*c__y*u__y
    """
    name = 'WaveEquation'

    def __init__(self, u='u', c='c', dim=3, time=True, mixed_form=False):
        self.u = u
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        t = Symbol('t')
        input_variables = {'x': x, 'y': y, 'z': z, 't': t}
        if self.dim == 1:
            input_variables.pop('y')
            input_variables.pop('z')
        elif self.dim == 2:
            input_variables.pop('z')
        if not self.time:
            input_variables.pop('t')
        assert type(u) == str, 'u needs to be string'
        u = Function(u)(*input_variables)
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)
        self.equations = {}
        if not self.mixed_form:
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['wave_equation'] = u.diff(t, 2) - c ** 2 * u.diff(x,
                2) - c ** 2 * u.diff(y, 2) - c ** 2 * u.diff(z, 2)
        elif self.mixed_form:
            u_x = Function('u_x')(*input_variables)
            u_y = Function('u_y')(*input_variables)
            if self.dim == 3:
                u_z = Function('u_z')(*input_variables)
            else:
                u_z = Number(0)
            if self.time:
                u_t = Function('u_t')(*input_variables)
            else:
                u_t = Number(0)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['wave_equation'] = u_t.diff(t) - c ** 2 * u_x.diff(x
                ) - c ** 2 * u_y.diff(y) - c ** 2 * u_z.diff(z)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_x'] = u.diff(x) - u_x
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_y'] = u.diff(y) - u_y
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_z'] = u.diff(z) - u_z
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_xy'] = u_x.diff(y) - u_y.diff(x)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_xz'] = u_x.diff(z) - u_z.diff(x)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_yz'] = u_y.diff(z) - u_z.diff(y)
            if self.dim == 2:
                self.equations.pop('compatibility_u_z')
                self.equations.pop('compatibility_u_xz')
                self.equations.pop('compatibility_u_yz')


class HelmholtzEquation(PDE):
    name = 'HelmholtzEquation'

    def __init__(self, u, k, dim=3, mixed_form=False):
        """
        Helmholtz equation

        Parameters
        ==========
        u : str
            The dependent variable.
        k : float, Sympy Symbol/Expr, str
            Wave number. If `k` is a str then it is
            converted to Sympy Function of form 'k(x,y,z,t)'.
            If 'k' is a Sympy Symbol or Expression then this
            is substituted into the equation.
        dim : int
            Dimension of the wave equation (1, 2, or 3). Default is 2.
        mixed_form: bool
        If True, use the mixed formulation of the Helmholtz equation.
        """
        self.u = u
        self.dim = dim
        self.mixed_form = mixed_form
        x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
        input_variables = {'x': x, 'y': y, 'z': z}
        if self.dim == 1:
            input_variables.pop('y')
            input_variables.pop('z')
        elif self.dim == 2:
            input_variables.pop('z')
        assert type(u) == str, 'u needs to be string'
        u = Function(u)(*input_variables)
        if type(k) is str:
            k = Function(k)(*input_variables)
        elif type(k) in [float, int]:
            k = Number(k)
        self.equations = {}
        if not self.mixed_form:
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['helmholtz'] = -(k ** 2 * u + u.diff(x, 2) + u.
                diff(y, 2) + u.diff(z, 2))
        elif self.mixed_form:
            u_x = Function('u_x')(*input_variables)
            u_y = Function('u_y')(*input_variables)
            if self.dim == 3:
                u_z = Function('u_z')(*input_variables)
            else:
                u_z = Number(0)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['helmholtz'] = -(k ** 2 * u + u_x.diff(x) + u_y.
                diff(y) + u_z.diff(z))
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_x'] = u.diff(x) - u_x
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_y'] = u.diff(y) - u_y
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_z'] = u.diff(z) - u_z
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_xy'] = u_x.diff(y) - u_y.diff(x)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_xz'] = u_x.diff(z) - u_z.diff(x)
            """Class Method: *.diff, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.equations['compatibility_u_yz'] = u_y.diff(z) - u_z.diff(y)
            if self.dim == 2:
                self.equations.pop('compatibility_u_z')
                self.equations.pop('compatibility_u_xz')
                self.equations.pop('compatibility_u_yz')
