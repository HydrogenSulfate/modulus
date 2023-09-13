import paddle
import logging
from typing import List, Dict, Tuple, Union
import functorch
import logging
from typing import Optional, Dict, Union, List
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key
from modulus.sym.manager import GraphManager
logger = logging.getLogger(__name__)


class DeepONetArch(Arch):
    """DeepONet

    Parameters
    ----------
    branch_net : Arch
        Branch net model. Output key should be variable "branch"
    trunk_net : Arch
        Trunk net model. Output key should be variable "trunk"
    output_keys : List[Key], optional
        Output variable keys, by default None
    detach_keys : List[Key], optional
    List of keys to detach gradients, by default []
    branch_dim : Union[None, int], optional
        Dimension of the branch encoding vector. If none, the model will use the
        variable trunk dimension. Should be set for 2D/3D models. By default None
    trunk_dim : Union[None, int], optional
        Dimension of the trunk encoding vector. If none, the model will use the
        variable trunk dimension. Should be set for 2D/3D models. By default None

    Note
    ----
    The branch and trunk net should ideally output to the same dimensionality, but if
    this is not the case the DeepO model will use a linear layer to match both branch/trunk
    dimensionality to (branch_dim + trunk_dim)/2. This vector will then be
    used for the final output multiplication.

    Note
    ----
    Higher dimension branch networks are supported. If the output is not a 1D vector the
    DeepO model will reshape for the final output multiplication.

    Note
    ----
    For more info on DeepONet refer to: https://arxiv.org/abs/1910.03193
    """

    def __init__(self, branch_net: Arch, trunk_net: Arch, output_keys: List
        [Key]=None, detach_keys: List[Key]=[], branch_dim: Union[None, int]
        =None, trunk_dim: Union[None, int]=None) ->None:
        super().__init__(input_keys=[], output_keys=output_keys,
            detach_keys=detach_keys)
        self.branch_net = branch_net
        self.branch_dim = branch_dim
        self.trunk_net = trunk_net
        self.trunk_dim = trunk_dim
        self.input_keys = (self.branch_net.input_keys + self.trunk_net.
            input_keys)
        self.input_key_dict = {str(var): var.size for var in self.input_keys}
        self.input_scales = {str(k): k.scale for k in self.input_keys}
        if self.trunk_dim is None:
            self.trunk_dim = sum(self.trunk_net.output_key_dict.values())
        if self.branch_dim is None:
            self.branch_dim = sum(self.branch_net.output_key_dict.values())
        self.deepo_dim = (self.trunk_dim + self.branch_dim) // 2
        out_features = sum(self.output_key_dict.values())
        if not self.trunk_dim == self.branch_dim:
            self.branch_linear = paddle.nn.Linear(in_features=self.
                branch_dim, out_features=self.deepo_dim, bias_attr=False)
            self.trunk_linear = paddle.nn.Linear(in_features=self.trunk_dim,
                out_features=self.deepo_dim, bias_attr=False)
        else:
            self.branch_linear = paddle.nn.Identity()
            self.trunk_linear = paddle.nn.Identity()
        self.output_linear = paddle.nn.Linear(in_features=self.deepo_dim,
            out_features=out_features, bias_attr=False)
        branch_slice_index = self.prepare_slice_index(self.input_key_dict,
            self.branch_net.input_key_dict.keys())
        self.register_buffer(name='branch_slice_index', tensor=
            branch_slice_index, persistable=False)
        trunk_slice_index = self.prepare_slice_index(self.input_key_dict,
            self.trunk_net.input_key_dict.keys())
        self.register_buffer(name='trunk_slice_index', tensor=
            trunk_slice_index, persistable=False)
        if not self.supports_func_arch:
            self.forward = self._dict_forward
            if GraphManager().func_arch:
                logger.warning(
                    f'The combination of branch_net ({type(self.branch_net)}) and trunk_net'
                     + f'({type(self.trunk_net)}) does not support FuncArch.')

    @property
    def supports_func_arch(self) ->bool:
        return (self.branch_net.supports_func_arch and self.trunk_net.
            supports_func_arch)

    def _tensor_forward(self, x: paddle.Tensor) ->paddle.Tensor:
        assert self.supports_func_arch, f'The combination of branch_net {type(self.branch_net)} and trunk_net ' + f'{type(self.trunk_net)} does not support FuncArch.'
        branch_x = self.slice_input(x, self.branch_slice_index, dim=-1)
        trunk_x = self.slice_input(x, self.trunk_slice_index, dim=-1)
        branch_output = self.branch_net._tensor_forward(branch_x)
        trunk_output = self.trunk_net._tensor_forward(trunk_x)
        if torch._C._functorch.is_gradtrackingtensor(trunk_output
            ) or torch._C._functorch.is_batchedtensor(trunk_output):
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            branch_output = branch_output.view(-1)
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            trunk_output = trunk_output.view(-1)
        else:
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            branch_output = branch_output.view(branch_output.shape[0], -1)
            """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            trunk_output = trunk_output.view(trunk_output.shape[0], -1)
        assert branch_output.shape[-1
            ] == self.branch_dim, f'Invalid feature dimension from branch net, expected {self.branch_dim} but found {branch_output.shape[-1]}'
        assert trunk_output.shape[-1
            ] == self.trunk_dim, f'Invalid feature dimension from trunk net, expected {self.trunk_dim} but found {trunk_output.shape[-1]}'
        branch_output = self.branch_linear(branch_output)
        trunk_output = self.trunk_linear(trunk_output)
        y = self.output_linear(branch_output * trunk_output)
        y = self.process_output(y, self.output_scales_tensor)
        return y

    def forward(self, in_vars: Dict[str, Tensor]) ->Dict[str, Tensor]:
        x = self.concat_input(in_vars, self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict, dim=-1)
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) ->Dict[str, Tensor]:
        branch_output = self.branch_net(in_vars)
        trunk_output = self.trunk_net(in_vars)
        branch_output = branch_output['branch']
        trunk_output = trunk_output['trunk']
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        branch_output = branch_output.reshape([branch_output.shape[0], -1])
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        trunk_output = trunk_output.reshape([trunk_output.shape[0], -1])
        assert branch_output.shape[-1
            ] == self.branch_dim, f'Invalid feature dimension from branch net, expected {self.branch_dim} but found {branch_output.shape[-1]}'
        assert trunk_output.shape[-1
            ] == self.trunk_dim, f'Invalid feature dimension from trunk net, expected {self.trunk_dim} but found {trunk_output.shape[-1]}'
        branch_output = self.branch_linear(branch_output)
        trunk_output = self.trunk_linear(trunk_output)
        out = self.output_linear(branch_output * trunk_output)
        return self.prepare_output(out, self.output_key_dict, dim=-1,
            output_scales=self.output_scales)
