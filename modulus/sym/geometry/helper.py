import numpy as np
import sympy
import itertools
from modulus.sym.utils.sympy import np_lambdify
from modulus.sym.constants import diff_str


def _concat_numpy_dict_list(numpy_dict_list):
    concat_variable = {}
    for key in numpy_dict_list[0].keys():
        concat_variable[key] = np.concatenate([x[key] for x in
            numpy_dict_list], axis=0)
    return concat_variable


def _sympy_sdf_to_sdf(sdf, dx=0.0001):
    sdf_inputs = list(set([str(x) for x in sdf.free_symbols]))
    fn_sdf = np_lambdify(sdf, sdf_inputs)

    def _sdf(fn_sdf, sdf_inputs, dx):

        def sdf(invar, params, compute_sdf_derivatives=False):
            inputs = {}
            for key, value in itertools.chain(invar.items(), params.items()):
                if key in sdf_inputs:
                    inputs[key] = value
            computed_sdf = fn_sdf(**inputs)
            outputs = {'sdf': computed_sdf}
            if compute_sdf_derivatives:
                for d in [x for x in invar.keys() if x in ['x', 'y', 'z']]:
                    if d in sdf_inputs:
                        inputs_plus = {**inputs}
                        inputs_plus[d] = inputs_plus[d] + dx / 2
                        computed_sdf_plus = fn_sdf(**inputs_plus)
                        inputs_minus = {**inputs}
                        inputs_minus[d] = inputs_minus[d] - dx / 2
                        computed_sdf_minus = fn_sdf(**inputs_minus)
                        outputs['sdf' + diff_str + d] = (computed_sdf_plus -
                            computed_sdf_minus) / dx
                    else:
                        outputs['sdf' + diff_str + d] = np.zeros_like(
                            computed_sdf)
            return outputs
        return sdf
    return _sdf(fn_sdf, sdf_inputs, dx)


def _sympy_criteria_to_criteria(criteria):
    criteria_inputs = list(set([str(x) for x in criteria.free_symbols]))
    fn_criteria = np_lambdify(criteria, criteria_inputs)

    def _criteria(fn_criteria, criteria_inputs):

        def criteria(invar, params):
            inputs = {}
            for key, value in itertools.chain(invar.items(), params.items()):
                if key in criteria_inputs:
                    inputs[key] = value
            return fn_criteria(**inputs)
        return criteria
    return _criteria(fn_criteria, criteria_inputs)


def _sympy_func_to_func(func):
    func_inputs = list(set([str(x) for x in func.free_symbols]))
    fn_func = np_lambdify(func, func_inputs)

    def _func(fn_func, func_inputs):

        def func(params):
            inputs = {}
            for key, value in params.items():
                if key in func_inputs:
                    inputs[key] = value
            return fn_func(**inputs)
        return func
    return _func(fn_func, func_inputs)
