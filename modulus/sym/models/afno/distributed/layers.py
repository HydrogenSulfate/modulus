import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
import math
import warnings
from modulus.sym.distributed.manager import DistributedManager
from modulus.sym.models.afno.distributed.mappings import copy_to_matmul_parallel_region
from modulus.sym.models.afno.distributed.mappings import reduce_from_matmul_parallel_region
from modulus.sym.models.afno.distributed.mappings import scatter_to_matmul_parallel_region
from modulus.sym.models.afno.distributed.mappings import gather_from_matmul_parallel_region
from modulus.sym.distributed.helpers import _transpose
from modulus.sym.distributed.helpers import pad_helper
from modulus.sym.distributed.helpers import truncate_helper


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.'
            , stacklevel=2)
    with paddle.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
        tensor.erfinv_()
        """Class Method: *.mul_, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        tensor = tensor * std * math.sqrt(2.0)
        tensor.add_(y=paddle.to_tensor(mean))
        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
    tensor: an n-dimensional `torch.Tensor`
    mean: the mean of the normal distribution
    std: the standard deviation of the normal distribution
    a: the minimum cutoff value
    b: the maximum cutoff value
    Examples:
    >>> w = torch.empty(3, 5)
    >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


>>>@torch.jit.script
def drop_path(x: paddle.Tensor, drop_prob: float=0.0, training: bool=False
    ) ->paddle.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape=shape, dtype=x.dtype)
    random_tensor.floor_()
    output = paddle.divide(x=x, y=paddle.to_tensor(keep_prob)) * random_tensor
    return output


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DistributedMLP(paddle.nn.Layer):

    def __init__(self, in_features, hidden_features=None, out_features=None,
        act_layer=paddle.nn.GELU, drop=0.0, input_is_matmul_parallel=False,
        output_is_matmul_parallel=False):
        super(DistributedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        comm_size = DistributedManager().group_size('model_parallel')
        assert hidden_features % comm_size == 0, 'Error, hidden_features needs to be divisible by matmul_parallel_size'
        hidden_features_local = hidden_features // comm_size
        out_83 = paddle.create_parameter(shape=paddle.ones(shape=[
            hidden_features_local, in_features, 1, 1]).shape, dtype=paddle.
            ones(shape=[hidden_features_local, in_features, 1, 1]).numpy().
            dtype, default_initializer=paddle.nn.initializer.Assign(paddle.
            ones(shape=[hidden_features_local, in_features, 1, 1])))
        out_83.stop_gradient = not True
        self.w1 = out_83
        out_84 = paddle.create_parameter(shape=paddle.zeros(shape=
            hidden_features_local).shape, dtype=paddle.zeros(shape=
            hidden_features_local).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(paddle.zeros(shape=
            hidden_features_local)))
        out_84.stop_gradient = not True
        self.b1 = out_84
        out_85 = paddle.create_parameter(shape=paddle.ones(shape=[
            out_features, hidden_features_local, 1, 1]).shape, dtype=paddle
            .ones(shape=[out_features, hidden_features_local, 1, 1]).numpy(
            ).dtype, default_initializer=paddle.nn.initializer.Assign(
            paddle.ones(shape=[out_features, hidden_features_local, 1, 1])))
        out_85.stop_gradient = not True
        self.w2 = out_85
        out_86 = paddle.create_parameter(shape=paddle.zeros(shape=
            out_features).shape, dtype=paddle.zeros(shape=out_features).
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (paddle.zeros(shape=out_features)))
        out_86.stop_gradient = not True
        self.b2 = out_86
        self.act = act_layer()
        self.drop = paddle.nn.Dropout(p=drop
            ) if drop > 0.0 else paddle.nn.Identity()
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.w1, std=0.02)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.b1)
        trunc_normal_(self.w2, std=0.02)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.b2)

    def forward(self, x):
        if self.input_is_matmul_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)
        x = copy_to_matmul_parallel_region(x)
        x = paddle.nn.functional.conv2d(x=x, weight=self.w1, bias=self.b1)
        x = self.act(x)
        x = self.drop(x)
        x = paddle.nn.functional.conv2d(x=x, weight=self.w2, bias=None)
        x = reduce_from_matmul_parallel_region(x)
        x = x + paddle.reshape(x=self.b2, shape=(1, -1, 1, 1))
        x = self.drop(x)
        if self.output_is_matmul_parallel:
            x = scatter_to_matmul_parallel_region(x, dim=1)
        return x


class DistributedPatchEmbed(paddle.nn.Layer):

    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3,
        embed_dim=768, input_is_matmul_parallel=False,
        output_is_matmul_parallel=True):
        super(DistributedPatchEmbed, self).__init__()
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel
        matmul_comm_size = DistributedManager().group_size('model_parallel')
        num_patches = img_size[1] // patch_size[1] * (img_size[0] //
            patch_size[0])
        self.img_size = img_size[0], img_size[1]
        self.patch_size = patch_size
        self.num_patches = num_patches
        if self.input_parallel:
            assert in_chans % matmul_comm_size == 0, 'Error, the in_chans needs to be divisible by matmul_parallel_size'
        if self.output_parallel:
            assert embed_dim % matmul_comm_size == 0, 'Error, the embed_dim needs to be divisible by matmul_parallel_size'
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim
        self.proj = paddle.nn.Conv2D(in_channels=in_chans, out_channels=
            out_chans_local, kernel_size=patch_size, stride=patch_size)
        self.proj.weight.is_shared_spatial = True
        self.proj.bias.is_shared_spatial = True

    def forward(self, x):
        if self.input_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)
        if self.output_parallel:
            x = copy_to_matmul_parallel_region(x)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1
            ], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_axis=2)
        return x


>>>@torch.jit.script
def compl_mul_add_fwd(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor
    ) ->paddle.Tensor:
    tmp = paddle.einsum('bkixys,kiot->stbkoxy', a, b)
    res = paddle.stack(x=[tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] +
        tmp[0, 1, ...]], axis=-1) + c
    return res


>>>@torch.jit.script
def compl_mul_add_fwd_c(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor
    ) ->paddle.Tensor:
    ac = paddle.as_complex(x=a)
    bc = paddle.as_complex(x=b)
    cc = paddle.as_complex(x=c)
    tmp = paddle.einsum('bkixy,kio->bkoxy', ac, bc)
    res = tmp + cc
    return paddle.as_real(x=res)


class DistributedAFNO2D(paddle.nn.Layer):

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
        hard_thresholding_fraction=1, hidden_size_factor=1,
        input_is_matmul_parallel=False, output_is_matmul_parallel=False):
        super(DistributedAFNO2D, self).__init__()
        assert hidden_size % num_blocks == 0, f'hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}'
        matmul_comm_size = DistributedManager().group_size('model_parallel')
        self.fft_handle = paddle.fft.rfft2
        self.ifft_handle = paddle.fft.irfft2
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        assert self.num_blocks % matmul_comm_size == 0, 'Error, num_blocks needs to be divisible by matmul_parallel_size'
        self.num_blocks_local = self.num_blocks // matmul_comm_size
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        use_complex_mult = False
        self.mult_handle = (compl_mul_add_fwd_c if use_complex_mult else
            compl_mul_add_fwd)
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel
        out_87 = paddle.create_parameter(shape=(self.scale * paddle.randn(
            shape=[self.num_blocks_local, self.block_size, self.block_size *
            self.hidden_size_factor, 2])).shape, dtype=(self.scale * paddle
            .randn(shape=[self.num_blocks_local, self.block_size, self.
            block_size * self.hidden_size_factor, 2])).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(self.scale *
            paddle.randn(shape=[self.num_blocks_local, self.block_size,
            self.block_size * self.hidden_size_factor, 2])))
        out_87.stop_gradient = not True
        self.w1 = out_87
        out_88 = paddle.create_parameter(shape=(self.scale * paddle.randn(
            shape=[self.num_blocks_local, self.block_size * self.
            hidden_size_factor, 1, 1, 2])).shape, dtype=(self.scale *
            paddle.randn(shape=[self.num_blocks_local, self.block_size *
            self.hidden_size_factor, 1, 1, 2])).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(self.scale *
            paddle.randn(shape=[self.num_blocks_local, self.block_size *
            self.hidden_size_factor, 1, 1, 2])))
        out_88.stop_gradient = not True
        self.b1 = out_88
        out_89 = paddle.create_parameter(shape=(self.scale * paddle.randn(
            shape=[self.num_blocks_local, self.block_size * self.
            hidden_size_factor, self.block_size, 2])).shape, dtype=(self.
            scale * paddle.randn(shape=[self.num_blocks_local, self.
            block_size * self.hidden_size_factor, self.block_size, 2])).
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (self.scale * paddle.randn(shape=[self.num_blocks_local, self.
            block_size * self.hidden_size_factor, self.block_size, 2])))
        out_89.stop_gradient = not True
        self.w2 = out_89
        out_90 = paddle.create_parameter(shape=(self.scale * paddle.randn(
            shape=[self.num_blocks_local, self.block_size, 1, 1, 2])).shape,
            dtype=(self.scale * paddle.randn(shape=[self.num_blocks_local,
            self.block_size, 1, 1, 2])).numpy().dtype, default_initializer=
            paddle.nn.initializer.Assign(self.scale * paddle.randn(shape=[
            self.num_blocks_local, self.block_size, 1, 1, 2])))
        out_90.stop_gradient = not True
        self.b2 = out_90
        self.w1.is_shared_spatial = True
        self.b1.is_shared_spatial = True
        self.w2.is_shared_spatial = True
        self.b2.is_shared_spatial = True

    def forward(self, x):
        if not self.input_is_matmul_parallel:
            x = scatter_to_matmul_parallel_region(x, dim=1)
        bias = x
        dtype = x.dtype
        x = x.astype(dtype='float32')
        B, C, H, W = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        x = self.fft_handle(x, (H, W), (-2, -1), 'ortho')
        """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        x = x.reshape([B, self.num_blocks_local, self.block_size, H, W // 2 + 1])
        x = paddle.as_real(x=x)
        o2 = paddle.zeros(shape=x.shape)
        o1 = paddle.nn.functional.relu(x=self.mult_handle(x[:, :, :,
            total_modes - kept_modes:total_modes + kept_modes, :kept_modes,
            :], self.w1, self.b1))
        o2[:, :, :, total_modes - kept_modes:total_modes + kept_modes, :
            kept_modes, :] = self.mult_handle(o1, self.w2, self.b2)
        x = paddle.nn.functional.softshrink(x=o2, threshold=self.
            sparsity_threshold)
        x = paddle.as_complex(x=x)
        x = x.reshape(B, C, H, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), 'ortho')
        x = x.astype(dtype) + bias
        if not self.output_is_matmul_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)
        return x
