import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
""" Continuous type constraints
"""
import logging
from typing import Dict, List, Union
import numpy as np
from modulus.sym.domain.constraint import Constraint
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.loss import Loss, PointwiseLossNorm
from modulus.sym.distributed import DistributedManager
from modulus.sym.utils.io.vtk import grid_to_vtk
from modulus.sym.dataset import Dataset, IterableDataset, DictGridDataset
logger = logging.getLogger(__name__)


class SupervisedGridConstraint(Constraint):
    """Data-driven grid field constraint

    Parameters
    ----------
    nodes : List[Node]
            List of Modulus Nodes to unroll graph with.
    dataset: Union[Dataset, IterableDataset]
        dataset which supplies invar and outvar examples
        Must be a subclass of Dataset or IterableDataset
    loss : Loss, optional
        Modulus `Loss` function, by default PointwiseLossNorm()
    batch_size : int, optional
        Batch size used when running constraint, must be specified if Dataset used
        Not used if IterableDataset used
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
        Not used if IterableDataset used
    drop_last : bool, optional
        Drop last mini-batch if dataset not fully divisible but batch_size, by default False
        Not used if IterableDataset used
    num_workers : int, optional
        Number of dataloader workers, by default 0
    """

    def __init__(self, nodes: List[Node], dataset: Union[Dataset,
        IterableDataset], loss: Loss=PointwiseLossNorm(), batch_size: int=
        None, shuffle: bool=True, drop_last: bool=True, num_workers: int=0):
        super().__init__(nodes=nodes, dataset=dataset, loss=loss,
            batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=num_workers)

    def save_batch(self, filename):
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar0 = {key: value for key, value in invar.items()}
        invar = Constraint._set_device(invar, device=self.place,
            requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.place)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=
            self.place)
        if hasattr(self.model, 'module'):
            modl = self.model.module
        else:
            modl = self.model
        pred_outvar = modl(invar)
        named_true_outvar = {('true_' + key): value for key, value in
            true_outvar.items()}
        named_pred_outvar = {('pred_' + key): value for key, value in
            pred_outvar.items()}
        save_var = {**{key: value for key, value in invar0.items()}, **
            named_true_outvar, **named_pred_outvar}
        save_var = {key: value.cpu().detach().numpy() for key, value in
            save_var.items()}
        model_parallel_rank = self.manager.group_rank('model_parallel'
            ) if self.manager.distributed else 0
        grid_to_vtk(save_var, filename + f'-{model_parallel_rank}')

    def load_data(self):
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        self._input_vars = Constraint._set_device(invar, device=self.place,
            requires_grad=True)
        self._target_vars = Constraint._set_device(true_outvar, device=self
            .place)
        self._lambda_weighting = Constraint._set_device(lambda_weighting,
            device=self.place)

    def load_data_static(self):
        if self._input_vars is None:
            self.load_data()
        else:
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            input_vars = Constraint._set_device(invar, device=self.place,
                requires_grad=False)
            target_vars = Constraint._set_device(true_outvar, device=self.place
                )
            lambda_weighting = Constraint._set_device(lambda_weighting,
                device=self.place)
            for key in input_vars.keys():
                self._input_vars[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                paddle.assign(target_vars[key], output=self._target_vars[key])
            for key in lambda_weighting.keys():
                paddle.assign(lambda_weighting[key], output=self.
                    _lambda_weighting[key])

    def forward(self):
        self._output_vars = self.model(self._input_vars)

    def loss(self, step: int) ->Dict[str, paddle.Tensor]:
        if self._output_vars is None:
            logger.warn('Calling loss without forward call')
            return {}
        losses = self._loss(self._input_vars, self._output_vars, self.
            _target_vars, self._lambda_weighting, step)
        return losses


class _DeepONetConstraint(Constraint):

    def __init__(self, nodes: List[Node], invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array], outvar: Dict[str, np.array],
        batch_size: int, lambda_weighting: Dict[str, np.array], loss: Loss,
        shuffle: bool, drop_last: bool, num_workers: int):
        self.dataset = DictGridDataset(invar=invar_branch, outvar=outvar,
            lambda_weighting=lambda_weighting)
        self.manager = DistributedManager()
        self.place = self.manager.place
        if not drop_last and self.manager.cuda_graphs:
            logger.info('drop_last must be true when using cuda graphs')
            drop_last = True
        self.dataloader = iter(Constraint.get_dataloader(dataset=self.
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=
            drop_last, num_workers=num_workers))
        self.model = Graph(nodes, Key.convert_list(invar_branch.keys()) +
            Key.convert_list(invar_trunk.keys()), Key.convert_list(outvar.
            keys()))
        self.model.to(self.place)
        if self.manager.distributed:
            s = paddle.device.cuda.Stream()
            s.wait_stream(paddle.device.cuda.current_stream())
            with torch.cuda.stream(s):
                self.model = torch.nn.parallel.DistributedDataParallel(self
                    .model, device_ids=[self.manager.local_rank],
                    output_device=self.place, broadcast_buffers=self.
                    manager.broadcast_buffers, find_unused_parameters=self.
                    manager.find_unused_parameters, process_group=self.
                    manager.group('data_parallel'))
            paddle.device.cuda.current_stream().wait_stream(s)
        self._input_names = Key.convert_list(self.dataset.invar_keys)
        self._output_names = Key.convert_list(self.dataset.outvar_keys)
        self._input_vars_branch = None
        self._target_vars = None
        self._lambda_weighting = None
        self._loss = loss.to(self.place)

    def save_batch(self, filename):
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar0 = {key: value for key, value in invar.items()}
        invar = Constraint._set_device(invar, device=self.place,
            requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.place)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=
            self.place)
        if hasattr(self.model, 'module'):
            modl = self.model.module
        else:
            modl = self.model
        pred_outvar = modl({**invar, **self._input_vars_trunk})
        named_lambda_weighting = {('lambda_' + key): value for key, value in
            lambda_weighting.items()}
        named_true_outvar = {('true_' + key): value for key, value in
            true_outvar.items()}
        named_pred_outvar = {('pred_' + key): value for key, value in
            pred_outvar.items()}
        save_var = {**{key: value for key, value in invar0.items()}, **
            named_true_outvar, **named_pred_outvar, **named_lambda_weighting}
        save_var = {key: value.cpu().detach().numpy() for key, value in
            save_var.items()}
        model_parallel_rank = self.manager.group_rank('model_parallel'
            ) if self.manager.distributed else 0
        np.savez_compressed(filename + f'.{model_parallel_rank}.npz', **
            save_var)

    def load_data(self):
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        self._input_vars_branch = Constraint._set_device(invar, device=self
            .place, requires_grad=True)
        self._target_vars = Constraint._set_device(true_outvar, device=self
            .place)
        self._lambda_weighting = Constraint._set_device(lambda_weighting,
            device=self.place)

    def load_data_static(self):
        if self._input_vars_branch is None:
            self.load_data()
        else:
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            input_vars = Constraint._set_device(invar, device=self.place,
                requires_grad=False)
            target_vars = Constraint._set_device(true_outvar, device=self.place
                )
            lambda_weighting = Constraint._set_device(lambda_weighting,
                device=self.place)
            for key in input_vars.keys():
                self._input_vars_branch[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                paddle.assign(target_vars[key], output=self._target_vars[key])
            for key in lambda_weighting.keys():
                paddle.assign(lambda_weighting[key], output=self.
                    _lambda_weighting[key])

    def forward(self):
        self._output_vars = self.model({**self._input_vars_branch, **self.
            _input_vars_trunk})


class DeepONetConstraint_Data(_DeepONetConstraint):

    def __init__(self, nodes: List[Node], invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array], outvar: Dict[str, np.array],
        batch_size: int, lambda_weighting: Dict[str, np.array]=None, loss:
        Loss=PointwiseLossNorm(), shuffle: bool=True, drop_last: bool=True,
        num_workers: int=0):
        super().__init__(nodes=nodes, invar_branch=invar_branch,
            invar_trunk=invar_trunk, outvar=outvar, batch_size=batch_size,
            lambda_weighting=lambda_weighting, loss=loss, shuffle=shuffle,
            drop_last=drop_last, num_workers=num_workers)
        self._input_vars_trunk = Constraint._set_device(invar_trunk, device
            =self.place, requires_grad=True)

    def loss(self, step: int):
        losses = self._loss(self._input_vars_trunk, self._output_vars, self
            ._target_vars, self._lambda_weighting, step)
        return losses


class DeepONetConstraint_Physics(_DeepONetConstraint):

    def __init__(self, nodes: List[Node], invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array], outvar: Dict[str, np.array],
        batch_size: int, lambda_weighting: Dict[str, np.array]=None, loss:
        Loss=PointwiseLossNorm(), shuffle: bool=True, drop_last: bool=True,
        num_workers: int=0, tile_trunk_input: bool=True):
        super().__init__(nodes=nodes, invar_branch=invar_branch,
            invar_trunk=invar_trunk, outvar=outvar, batch_size=batch_size,
            lambda_weighting=lambda_weighting, loss=loss, shuffle=shuffle,
            drop_last=drop_last, num_workers=num_workers)
        if tile_trunk_input:
            for k, v in invar_trunk.items():
                invar_trunk[k] = np.tile(v, (batch_size, 1))
        self._input_vars_trunk = Constraint._set_device(invar_trunk, device
            =self.place, requires_grad=True)

    def loss(self, step: int):
        target_vars = {k: paddle.reshape(x=v, shape=(-1, 1)) for k, v in
            self._target_vars.items()}
        lambda_weighting = {k: paddle.reshape(x=v, shape=(-1, 1)) for k, v in
            self._lambda_weighting.items()}
        losses = self._loss(self._input_vars_trunk, self._output_vars,
            target_vars, lambda_weighting, step)
        return losses
