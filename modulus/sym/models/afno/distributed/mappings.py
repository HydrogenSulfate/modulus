import paddle
import types
from modulus.sym.distributed.manager import DistributedManager
from modulus.sym.distributed.helpers import split_tensor_along_dim
from modulus.sym.distributed.helpers import _reduce
from modulus.sym.distributed.helpers import _split
from modulus.sym.distributed.helpers import _gather


class _CopyToMatmulParallelRegion(paddle.autograd.PyLayer):
    """Pass the input to the matmul parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=DistributedManager().group(
            'model_parallel'))


class _ReduceFromMatmulParallelRegion(paddle.autograd.PyLayer):
    """All-reduce the input from the matmul parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToMatmulParallelRegion(paddle.autograd.PyLayer):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _split(input_, dim_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, group=DistributedManager().
            group('model_parallel')), None


class _GatherFromMatmulParallelRegion(paddle.autograd.PyLayer):
    """Gather the input from matmul parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, group=DistributedManager().
            group('model_parallel')), None


class _GatherWithinMatmulParallelRegion(paddle.autograd.PyLayer):
    """Gather the input from matmul parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=DistributedManager().group(
            'model_parallel'))

    @staticmethod
    def backward(ctx, grad_output):
        red = _reduce(grad_output, group=DistributedManager().group(
            'model_parallel'))
        return _split(red, ctx.dim, group=DistributedManager().group(
            'model_parallel')), None


def copy_to_matmul_parallel_region(input_):
    return _CopyToMatmulParallelRegion.apply(input_)


def reduce_from_matmul_parallel_region(input_):
    return _ReduceFromMatmulParallelRegion.apply(input_)


def scatter_to_matmul_parallel_region(input_, dim):
    return _ScatterToMatmulParallelRegion.apply(input_, dim)


def gather_from_matmul_parallel_region(input_, dim):
    return _GatherFromMatmulParallelRegion.apply(input_, dim)


def gather_within_matmul_parallel_region(input_, dim):
    return _GatherWithinMatmulParallelRegion.apply(input_, dim)
