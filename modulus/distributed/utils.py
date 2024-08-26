# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO this also needs more docstrings
from __future__ import annotations

from typing import List, Optional

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F

from .manager import DistributedManager


def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    # treat trivial case first
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


# def get_memory_format(tensor):
#     """Gets format for tensor"""
#     if tensor.is_contiguous():
#         return paddle.channels_last
#     else:
#         return paddle.contiguous_format


def pad_helper(tensor, dim, new_size, mode="zero"):
    """Util for padding tensors"""
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode="constant", value=0.0)

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad[lhs_slice] = paddle.flip(
            paddle.conj(tensor_pad[rhs_slice]), axis=[dim]
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    """Util for truncating"""
    # input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[output_slice].contiguous()

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    if dim >= tensor.dim():
        raise ValueError(
            f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
        )
    if tensor.shape[dim] < num_chunks:
        raise ValueError(
            "Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
        {num_chunks} chunks. Empty slices are currently not supported."
        )

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = paddle.split(tensor, sections, axis=dim)

    return tensor_list


@paddle.no_grad()
def reduce_loss(loss: float, dst_rank: int = 0, mean: bool = True):  # pragma: no cover
    """Reduces loss from all processes to destination rank for logging.

    Parameters
    ----------
    loss : float
        loss value
    dst_rank : int, Optional
        destination rank to redce to, by default 0.
    mean : bool, Optional
        Calculate the mean of the losses gathered, by default True.

    Raises
    ------
    Exception
        If DistributedManager has yet to be initialized
    """
    if not DistributedManager.is_initialized():
        raise Exception(
            "Distributed manager should be initialized when using reduce_loss"
        )

    distmng = DistributedManager()
    loss = paddle.to_tensor([loss]).to(distmng.device)

    # For serial runs, just return the current loss!
    if distmng.world_size == 1:
        return float(loss)

    op = (
        paddle.distributed.ReduceOp.SUM if not mean else paddle.distributed.ReduceOp.AVG
    )
    paddle.distributed.reduce(loss, dst_rank, op, group=None)

    # Return loss if dst_rank, None otherwise
    if distmng.rank == dst_rank:
        return float(loss.cpu())
    else:
        return None


# distributed primitives
def distributed_transpose(tensor, dim0, dim1, group=None, async_op=False):
    """Perform distributed transpose of tensor to switch sharding dimension"""
    # get input format
    # input_format = get_memory_format(tensor)

    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    x_send = [y.contiguous() for y in paddle.split(tensor, comm_size, axis=dim0)]
    x_recv = [paddle.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    req = dist.alltoall(x_recv, x_send, group=group, sync_op=not async_op)

    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):  # pragma: no cover
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce, use_fp32 only relevant for lower precisions
    # if input is already in double precision, nothing changes
    if use_fp32 and (input_.element_size() < 4) and input_.is_floating_point():
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.astype(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def _split(input_, dim_, group=None):  # pragma: no cover
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    # input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: paddle.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def all_gather_v_wrapper(
    tensor: paddle.Tensor,
    sizes: Optional[List[int]] = None,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "paddle.Tensor"
        local tensor on each rank
    sizes : List[int], optional
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank, by default None
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        full global tensor, valid on each rank
    """

    comm_size = dist.get_world_size(group=group)

    if (sizes is not None) and (len(sizes) != comm_size):
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    if comm_size == 1:
        return tensor

    tensor_shape = list(tensor.shape)
    # tensor_format = get_memory_format(tensor)

    if sizes is not None:
        tensor_list = [None] * comm_size

        for src in range(comm_size):
            tensor_shape[dim] = sizes[src]
            tensor_list[src] = paddle.empty(
                tensor_shape,
                dtype=tensor.dtype,
            )
    else:
        # assume equal shape on all ranks
        tensor_list = [paddle.empty_like(tensor) for _ in range(comm_size)]

    dist.all_gather(tensor_list, tensor, group=group)

    output = paddle.concat(tensor_list, axis=dim).contiguous()

    return output


def all_gather_v_bwd_wrapper(
    tensor: paddle.Tensor,
    sizes: List[int],
    dim: int = 0,
    use_fp32: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements a distributed AllReduceV primitive. It is based
    on the idea of a single global tensor which which can be distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes different global tensors of the same shape on each
    rank. It then re-distributes chunks of all these tensors such that each rank
    receives all corresponding parts of a global tensor. Each rank then sums up
    the chunks after receiving it. By design, this primitive thus implements the
    backward pass of the "all_gather_v" primitive. In this case, the result would
    be a single global gradient tensor distributed onto different ranks.

    Parameters
    ----------
    tensor : paddle.Tensor
        global tensor on each rank (different one on each rank)
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        local tensor, i.e. result of reduction of all corresponding chunks
        from all global tensors for each rank separately
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = sizes[rank]
    tmp = [
        paddle.empty(
            tensor_shape,
            dtype=tensor.dtype,
        )
        for _ in range(comm_size)
    ]
    scatter_list = list(paddle.split(tensor, sizes, axis=dim))
    scatter_list = [t.contiguous() for t in scatter_list]

    dist.alltoall(tmp, scatter_list, group=group)
    stack_dim = tensor.dim()
    tmp = paddle.stack(tmp, axis=stack_dim)

    if use_fp32 and (tmp.element_size() < 4) and tmp.is_floating_point():
        # cast to float before sum and return float, then cast back
        output = tmp.sum(axis=stack_dim, dtype=paddle.float32)
        output = output.to(dtype=tensor.dtype)
    else:
        # else: just do sum in native dtype
        output = tmp.sum(axis=stack_dim)

    return output


def gather_v_wrapper(
    tensor: paddle.Tensor,
    sizes: List[int],
    dim: int = 0,
    dst: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements a distributed GatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank.

    Parameters
    ----------
    tensor : paddle.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the
        operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        full global tensor, valid on destination rank
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if not (0 <= dst < comm_size):
        raise ValueError()
    if tensor.size(dim) != sizes[rank]:
        raise ValueError()

    if comm_size == 1:
        return tensor

    tensor_shape = list(tensor.shape)
    x_recv = [None] * comm_size
    x_send = [None] * comm_size

    for r in range(comm_size):
        if rank == dst:
            tensor_shape[dim] = sizes[r]
        else:
            tensor_shape[dim] = 0

        x_recv[r] = paddle.empty(
            tensor_shape,
            dtype=tensor.dtype,
        )

        if r == dst:
            x_send[r] = tensor
        else:
            tensor_shape[dim] = 0
            x_send[r] = paddle.empty(
                tensor_shape,
                dtype=tensor.dtype,
            )

    dist.alltoall(x_recv, x_send, group=group)

    # TODO: clean gather/scatter and some examples up
    # main question is around whether e.g. gather returns
    # None for rank != dst or an empty dummy or an dummy
    # containing meta-information like dtype/etc..
    if rank != dst:
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r]
            x_recv[r] = paddle.empty(
                tensor_shape,
                dtype=tensor.dtype,
            )

    output = paddle.concat(x_recv, axis=dim)

    return output


def scatter_v_wrapper(
    tensor: paddle.Tensor,
    sizes: List[int],
    dim: int = 0,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements a distributed ScatterV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank.

    Parameters
    ----------
    tensor : paddle.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        corresponding local part of the global tensor on each rank
    """
    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dist.get_rank(group=group) == 0 and dim >= tensor.dim():
        raise ValueError()
    if not (0 <= src < comm_size):
        raise ValueError()

    # alltoall is already alltoall_v, use empty tensors to "mask"-out irrelevant parts
    tensor_shape = list(tensor.shape)
    x_send = [None] * comm_size
    x_recv = [None] * comm_size
    if rank == src:
        scatter_list = paddle.split(tensor, sizes, axis=dim)
        scatter_list = [t.contiguous() for t in scatter_list]
        x_send = scatter_list
    else:
        for r in range(comm_size):
            tensor_shape[dim] = 0
            x_send[r] = paddle.empty(tensor_shape, dtype=tensor.dtype)

    for r in range(comm_size):
        if r == src:
            tensor_shape[dim] = sizes[rank]
        else:
            tensor_shape[dim] = 0
        x_recv[r] = paddle.empty(tensor_shape, dtype=tensor.dtype)

    dist.alltoall(x_recv, x_send, group=group)

    return x_recv[src]


def indexed_alltoall_v_wrapper(
    tensor: paddle.Tensor,
    indices: List[paddle.Tensor],
    sizes: List[List[int]],
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : paddle.Tensor
        local part of global tensor on each rank
    indices : List[paddle.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    x_send = [tensor[idx] for idx in indices]
    x_recv = [None] * comm_size
    tensor_shape = list(tensor.shape)
    for r in range(comm_size):
        tensor_shape[dim] = sizes[r][rank]
        x_recv[r] = paddle.empty(
            tensor_shape,
            dtype=tensor.dtype,
        )

    dist.alltoall(x_recv, x_send, group=group)

    tensor_to_recv = paddle.concat(x_recv, axis=dim)

    return tensor_to_recv


def indexed_alltoall_v_wrapper_bwd(
    tensor: paddle.Tensor,
    indices: List[paddle.Tensor],
    sizes: List[List[int]],
    tensor_size_along_dim: int,
    use_fp32: bool = True,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : paddle.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[paddle.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    tensor_shape = list(tensor.shape)

    # scatter gradients, roles reversed compared to forward pass
    # recv_sizes in forward pass
    recv_sizes = [sizes[i][rank] for i in range(comm_size)]
    # send_sizes in forward pass
    send_sizes = [sizes[rank][i] for i in range(comm_size)]

    x_send = paddle.split(tensor, recv_sizes, axis=dim)
    x_send = [t.contiguous() for t in x_send]
    x_recv = [None] * comm_size
    for r in range(comm_size):
        tensor_shape[dim] = send_sizes[r]
        x_recv[r] = paddle.empty(
            tensor_shape,
            dtype=tensor.dtype,
        )

    dist.alltoall(x_recv, x_send, group=group)

    tensor_to_recv = paddle.concat(x_recv, axis=dim)

    # sum up gathered gradients and taking
    # care of precision handling as specified
    # by boolean flag
    indices = paddle.concat(indices, axis=0)
    tensor_shape[dim] = tensor_size_along_dim
    if use_fp32 and (tensor.element_size() < 4) and tensor.is_floating_point():
        out = paddle.zeros(
            tensor_shape,
            dtype=paddle.float32,
        )
        tensor_to_recv = tensor_to_recv.to(dtype=paddle.float32)
    else:
        out = paddle.zeros(
            tensor_shape,
            dtype=tensor.dtype,
        )

    out.index_add_(source=tensor_to_recv, index=indices, axis=dim)

    if out.dtype != tensor.dtype:
        out = out.to(tensor.dtype)

    return out


def indexed_all_to_all_v_wrapper(
    tensor: paddle.Tensor,
    indices: List[paddle.Tensor],
    sizes: List[List[int]],
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : paddle.Tensor
        local part of global tensor on each rank
    indices : List[paddle.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    x_send = [tensor[idx] for idx in indices]
    x_recv = [None] * comm_size
    tensor_shape = list(tensor.shape)
    for r in range(comm_size):
        tensor_shape[dim] = sizes[r][rank]
        x_recv[r] = paddle.empty(
            tensor_shape,
            dtype=tensor.dtype,
        ).to(device=tensor.place)

    dist.alltoall(x_recv, x_send, group=group)

    tensor_to_recv = paddle.concat(x_recv, axis=dim)

    return tensor_to_recv


def indexed_all_to_all_v_wrapper_bwd(
    tensor: paddle.Tensor,
    indices: List[paddle.Tensor],
    sizes: List[List[int]],
    tensor_size_along_dim: int,
    use_fp32: bool = True,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> paddle.Tensor:  # pragma: no cover
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : paddle.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[paddle.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify reduction taking place at least in FP32 precision, by default True
        only acts on floating point inputs in lower precision
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    paddle.Tensor
        result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    tensor_shape = list(tensor.shape)

    # scatter gradients, roles reversed compared to forward pass
    # recv_sizes in forward pass
    recv_sizes = [sizes[i][rank] for i in range(comm_size)]
    # send_sizes in forward pass
    send_sizes = [sizes[rank][i] for i in range(comm_size)]

    x_send = paddle.split(tensor, recv_sizes, axis=dim)
    x_send = [t.contiguous() for t in x_send]
    x_recv = [None] * comm_size
    for r in range(comm_size):
        tensor_shape[dim] = send_sizes[r]
        x_recv[r] = paddle.empty(
            tensor_shape,
            dtype=tensor.dtype,
        ).to(device=tensor.place)

    dist.alltoall(x_recv, x_send, group=group)

    tensor_to_recv = paddle.concat(x_recv, axis=dim)

    # sum up gathered gradients and taking
    # care of precision handling as specified
    # by boolean flag
    indices = paddle.concat(indices, axis=0)
    tensor_shape[dim] = tensor_size_along_dim
    if use_fp32 and (tensor.dtype.itemsize < 4) and tensor.dtype.is_floating_point:
        out = paddle.zeros(
            tensor_shape,
            dtype=paddle.float32,
        ).to(device=tensor.place)
        tensor_to_recv = tensor_to_recv.to(dtype=paddle.float32)
    else:
        out = paddle.zeros(
            tensor_shape,
            dtype=tensor.dtype,
        ).to(device=tensor.place)

    out.index_add_(source=tensor_to_recv, index=indices, axis=dim)

    if out.dtype != tensor.dtype:
        out = out.to(tensor.dtype)

    return out


def mark_module_as_shared(
    module: nn.Layer,
    process_group: Optional[str],
    recurse: bool = True,
    use_fp32_reduction: bool = True,
) -> nn.Layer:
    """
    Helper function to mark parameters of a module as being shared
    across ranks by attaching gradient hooks to the corresponding tensors.

    Parameters
    ----------
    module : nn.Layer
        Pypaddle module which is to be marked as having shared parameters.
    process_group : str | None
        str indicating process_group which contains ranks across which
        the module's parameters are shared. If passed as None, will default
        to the world group.
    recurse : bool, default=True
        Flag indicating whether the module's parameters are traversed in
        a recursive fashion, i.e. whether sub-modules are also considered
        as having shared parameters.
    use_fp32_reduction : bool, default=True
        Flag indicating whether the reduction for accumulating gradients
        will be done in at least FP32 or the native datatype.
    """

    group = DistributedManager().group(process_group)
    handle_key = "_shared_weight_dist_hook"

    def hook(grad: paddle.Tensor) -> paddle.Tensor:
        # the documentation states that
        # "The hook should not modify its argument, but it can optionally return a new gradient
        #  which will be used in place of grad."
        # as all_reduce is an in-place operation, need to copy gradient
        grad = _reduce(grad.clone(), group=group, use_fp32=use_fp32_reduction)
        return grad

    def hook_post_accum(param: paddle.Tensor) -> None:
        # the documentation states that
        # "Note that, unlike other autograd hooks, this hook operates on the tensor that requires grad
        #  and not the grad itself. The hook can in-place modify and access its Tensor argument,
        # including its .grad field."
        param.grad = _reduce(param.grad, group=group, use_fp32=use_fp32_reduction)

    for name, param in module.named_parameters(recurse=recurse):
        error_msg = f"Parameter {name} already marked as having shared weights, can't mark it again!"
        if hasattr(param, handle_key):
            raise RuntimeError(error_msg)
        if True:
            handle = param.register_hook(hook)
        # else:
        #     handle = param.register_post_accumulate_grad_hook(hook_post_accum)
        setattr(param, handle_key, handle)

    return module


def unmark_module_as_shared(
    module: nn.Layer,
    recurse: bool = True,
) -> nn.Layer:
    """
    Helper function to unmark parameters of a module as being shared
    across ranks by removing attached gradient hooks.

    Parameters
    ----------
    module : nn.Layer
        Pypaddle module which is to be unmarked as having shared parameters.
    recurse : bool, default=True
        Flag indicating whether the module's parameters are traversed in
        a recursive fashion, i.e. whether sub-modules are also considered
        as having shared parameters.
    """
    handle_key = "_shared_weight_dist_hook"
    for name, param in module.named_parameters(recurse=recurse):
        error_msg = (
            f"Parameter {name} NOT marked as having shared weights, can't unmark it!"
        )
        if not hasattr(param, handle_key):
            raise RuntimeError(error_msg)
        handle = getattr(param, handle_key)
        handle.remove()
        delattr(param, handle_key)

    return module
