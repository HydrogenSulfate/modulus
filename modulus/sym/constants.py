import paddle
"""
constant values used by Modulus
"""
import numpy as np
diff_str: str = '__'


def diff(y: str, x: str, degree: int=1) ->str:
    return diff_str.join([y] + degree * [x])


tf_dt = 'float32'
np_dt = np.float32
TF_SUMMARY = False
JIT_PYTORCH_VERSION = '2.1.0a0+4136153'
NO_OP_SCALE = 0.0, 1.0
NO_OP_NORM = -1.0, 1.0
