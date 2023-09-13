import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle


def get_memory_format(tensor):
    if True:
       return torch.channels_last
    else:
       return torch.contiguous_format


def pad_helper(tensor, dim, new_size, mode='zero'):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [(0) for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = paddle.nn.functional.pad(tensor, output_shape, mode=
        'constant', value=0.0)
    if mode == 'conj':
        lhs_slice = [(slice(0, x) if idx != dim else slice(orig_size,
            new_size)) for idx, x in enumerate(tensor.shape)]
        rhs_slice = [(slice(0, x) if idx != dim else slice(1, output_shape[
            1] + 1)) for idx, x in enumerate(tensor.shape)]
        tensor_pad[lhs_slice] = paddle.flip(x=paddle.conj(x=tensor_pad[
            rhs_slice]), axis=[dim])
    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [(slice(0, x) if idx != dim else slice(0, new_size)) for
        idx, x in enumerate(tensor.shape)]
    tensor_trunc = tensor[output_slice]
    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(
        ), f'Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}'
    assert tensor.shape[dim
        ] % num_chunks == 0, f'Error, cannot split dim {dim} evenly. Dim size is                                                   {tensor.shape[dim]} and requested numnber of splits is {num_chunks}'
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = paddle.split(x=tensor, num_or_sections=tensor.shape[dim] //
        chunk_size, axis=dim)
    return tensor_list


def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    input_format = get_memory_format(tensor)
    comm_size = paddle.distributed.get_world_size(group=group)
    split_size = tensor.shape[dim0] // comm_size
    x_send = [y for y in paddle.split(x=tensor, num_or_sections=tensor.
        shape[dim0] // split_size, axis=dim0)]
    x_recv = [paddle.empty_like(x=x_send[0]) for _ in range(comm_size)]
    req = paddle.distributed.alltoall(out_tensor_list=x_recv,
        in_tensor_list=x_send, group=group)
    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""
    if paddle.distributed.get_world_size(group=group) == 1:
        return input_
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.astype(dtype='float32')
        paddle.distributed.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        paddle.distributed.all_reduce(input_, group=group)
    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    input_format = get_memory_format(input_)
    comm_size = paddle.distributed.get_world_size(group=group)
    if comm_size == 1:
        return input_
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    rank = paddle.distributed.get_rank(group=group)
    output = input_list[rank]
    return output


def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    input_format = get_memory_format(input_)
    comm_size = paddle.distributed.get_world_size(group=group)
    if comm_size == 1:
        return input_
    assert dim_ < input_.dim(
        ), f'Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions.'
    comm_rank = paddle.distributed.get_rank(group=group)
    tensor_list = [paddle.empty_like(x=input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    paddle.distributed.all_gather(tensor_list, input_, group=group)
    output = paddle.concat(x=tensor_list, axis=dim_)
    return output
