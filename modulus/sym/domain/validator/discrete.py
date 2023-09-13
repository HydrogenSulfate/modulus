import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
from typing import Dict, List
import numpy as np
from modulus.sym.domain.validator import Validator
from modulus.sym.domain.constraint import Constraint
from modulus.sym.utils.io.vtk import grid_to_vtk
from modulus.sym.utils.io import GridValidatorPlotter, DeepONetValidatorPlotter
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.constants import TF_SUMMARY
from modulus.sym.distributed import DistributedManager
from modulus.sym.dataset import Dataset, DictGridDataset


class GridValidator(Validator):
    """Data-driven grid field validator

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    dataset: Dataset
        dataset which contains invar and true outvar examples
    batch_size : int, optional
            Batch size used when running validation, by default 100
    plotter : GridValidatorPlotter
        Modulus plotter for showing results in tensorboard.
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    num_workers : int, optional
        Number of dataloader workers, by default 0
    """

    def __init__(self, nodes: List[Node], dataset: Dataset, batch_size: int
        =100, plotter: GridValidatorPlotter=None, requires_grad: bool=False,
        num_workers: int=0):
        self.dataset = dataset
        self.dataloader = Constraint.get_dataloader(dataset=self.dataset,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, distributed=False, infinite=False)
        self.model = Graph(nodes, Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys))
        self.manager = DistributedManager()
        self.place = self.manager.place
        self.model.to(self.place)
        self.stop_gradient = not requires_grad
        self.forward = (self.forward_grad if requires_grad else self.
            forward_nograd)
        self.plotter = plotter

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.
            dataloader):
            invar = Constraint._set_device(invar0, device=self.place,
                requires_grad=not self.stop_gradient)
            true_outvar = Constraint._set_device(true_outvar0, device=self.
                place, requires_grad=not self.stop_gradient)
            pred_outvar = self.forward(invar)
            invar_cpu = {key: (value + [invar[key].cpu().detach()]) for key,
                value in invar_cpu.items()}
            true_outvar_cpu = {key: (value + [true_outvar[key].cpu().detach
                ()]) for key, value in true_outvar_cpu.items()}
            pred_outvar_cpu = {key: (value + [pred_outvar[key].cpu().detach
                ()]) for key, value in pred_outvar_cpu.items()}
        invar_cpu = {key: paddle.concat(x=value) for key, value in
            invar_cpu.items()}
        true_outvar_cpu = {key: paddle.concat(x=value) for key, value in
            true_outvar_cpu.items()}
        pred_outvar_cpu = {key: paddle.concat(x=value) for key, value in
            pred_outvar_cpu.items()}
        losses = GridValidator._l2_relative_error(true_outvar_cpu,
            pred_outvar_cpu)
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}
        named_true_outvar = {('true_' + k): v for k, v in true_outvar.items()}
        named_pred_outvar = {('pred_' + k): v for k, v in pred_outvar.items()}
        if 'np' in save_filetypes:
            np.savez(results_dir + name, {**invar, **named_true_outvar, **
                named_pred_outvar})
        if 'vtk' in save_filetypes:
            grid_to_vtk({**invar, **named_true_outvar, **named_pred_outvar},
                results_dir + name)
        if self.plotter is not None:
            self.plotter._add_figures('Validators', name, results_dir,
                writer, step, invar, true_outvar, pred_outvar)
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar('val/' + name + '/' + k, loss, step,
                    )
            else:
                writer.add_scalar('Validators/' + name + '/' + k, loss,
                    step)
        return losses


class _DeepONet_Validator(Validator):

    def __init__(self, nodes: List[Node], invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array], true_outvar: Dict[str, np.array],
        batch_size: int, plotter: DeepONetValidatorPlotter, requires_grad: bool
        ):
        self.dataset = DictGridDataset(invar=invar_branch, outvar=true_outvar)
        self.dataloader = Constraint.get_dataloader(dataset=self.dataset,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0, distributed=False, infinite=False)
        self.model = Graph(nodes, Key.convert_list(invar_branch.keys()) +
            Key.convert_list(invar_trunk.keys()), Key.convert_list(
            true_outvar.keys()))
        self.manager = DistributedManager()
        self.place = self.manager.place
        self.model.to(self.place)
        self.stop_gradient = not requires_grad
        self.forward = (self.forward_grad if requires_grad else self.
            forward_nograd)
        self.plotter = plotter


class DeepONet_Physics_Validator(_DeepONet_Validator):
    """
    DeepONet Validator
    """

    def __init__(self, nodes: List[Node], invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array], true_outvar: Dict[str, np.array],
        batch_size: int=100, plotter: DeepONetValidatorPlotter=None,
        requires_grad: bool=False, tile_trunk_input: bool=True):
        super().__init__(nodes=nodes, invar_branch=invar_branch,
            invar_trunk=invar_trunk, true_outvar=true_outvar, batch_size=
            batch_size, plotter=plotter, requires_grad=requires_grad)
        if tile_trunk_input:
            for k, v in invar_trunk.items():
                invar_trunk[k] = np.tile(v, (batch_size, 1))
        self.invar_trunk = invar_trunk
        self.batch_size = batch_size

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        invar_trunk_gpu = Constraint._set_device(self.invar_trunk, device=
            self.place, requires_grad=not self.stop_gradient)
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.
            dataloader):
            invar = Constraint._set_device(invar0, device=self.place,
                requires_grad=not self.stop_gradient)
            true_outvar = Constraint._set_device(true_outvar0, device=self.
                place, requires_grad=not self.stop_gradient)
            pred_outvar = self.forward({**invar, **invar_trunk_gpu})
            invar_cpu = {key: (value + [invar[key].cpu().detach()]) for key,
                value in invar_cpu.items()}
            true_outvar_cpu = {key: (value + [true_outvar[key].cpu().detach
                ()]) for key, value in true_outvar_cpu.items()}
            pred_outvar_cpu = {key: (value + [pred_outvar[key].cpu().detach
                ()]) for key, value in pred_outvar_cpu.items()}
        invar_cpu = {key: paddle.concat(x=value) for key, value in
            invar_cpu.items()}
        true_outvar_cpu = {key: paddle.concat(x=value) for key, value in
            true_outvar_cpu.items()}
        pred_outvar_cpu = {key: paddle.concat(x=value) for key, value in
            pred_outvar_cpu.items()}
        losses = DeepONet_Physics_Validator._l2_relative_error(true_outvar_cpu,
            pred_outvar_cpu)
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}
        named_true_outvar = {('true_' + k): v for k, v in true_outvar.items()}
        named_pred_outvar = {('pred_' + k): v for k, v in pred_outvar.items()}
        if 'np' in save_filetypes:
            np.savez(results_dir + name, {**invar, **named_true_outvar, **
                named_pred_outvar})
        ndim = next(iter(self.invar_trunk.values())).shape[-1]
        invar_plotter = dict()
        true_outvar_plotter = dict()
        pred_outvar_plotter = dict()
        for k, v in self.invar_trunk.items():
            invar_plotter[k] = self.invar_trunk[k].reshape((self.batch_size,
                -1, ndim))
        for k, v in true_outvar.items():
            true_outvar_plotter[k] = true_outvar[k].reshape((self.
                batch_size, -1))
        for k, v in pred_outvar.items():
            pred_outvar_plotter[k] = pred_outvar[k].reshape((self.
                batch_size, -1))
        if self.plotter is not None:
            self.plotter._add_figures('Validators', name, results_dir,
                writer, step, invar_plotter, true_outvar_plotter,
                pred_outvar_plotter)
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar('val/' + name + '/' + k, loss, step,
                    )
            else:
                writer.add_scalar('Validators/' + name + '/' + k, loss,
                    step)
        return losses

    @staticmethod
    def _l2_relative_error(true_var, pred_var):
        new_var = {}
        for key in true_var.keys():
            new_var['l2_relative_error_' + str(key)] = paddle.sqrt(x=paddle
                .mean(x=paddle.square(x=paddle.reshape(x=true_var[key],
                shape=(-1, 1)) - pred_var[key])) / paddle.var(x=true_var[key]))
        return new_var


class DeepONet_Data_Validator(_DeepONet_Validator):
    """
    DeepONet Validator
    """

    def __init__(self, nodes: List[Node], invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array], true_outvar: Dict[str, np.array],
        batch_size: int=100, plotter: DeepONetValidatorPlotter=None,
        requires_grad: bool=False):
        super().__init__(nodes=nodes, invar_branch=invar_branch,
            invar_trunk=invar_trunk, true_outvar=true_outvar, batch_size=
            batch_size, plotter=plotter, requires_grad=requires_grad)
        self.invar_trunk_plotter = dict()
        ndim = next(iter(invar_trunk.values())).shape[-1]
        for k, v in invar_trunk.items():
            self.invar_trunk_plotter[k] = np.tile(v, (batch_size, 1)).reshape((
                batch_size, -1, ndim))
        self.invar_trunk = invar_trunk

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        invar_trunk_gpu = Constraint._set_device(self.invar_trunk, device=
            self.place, requires_grad=not self.stop_gradient)
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.
            dataloader):
            invar = Constraint._set_device(invar0, device=self.place,
                requires_grad=not self.stop_gradient)
            true_outvar = Constraint._set_device(true_outvar0, device=self.
                place, requires_grad=not self.stop_gradient)
            pred_outvar = self.forward({**invar, **invar_trunk_gpu})
            invar_cpu = {key: (value + [invar[key].cpu().detach()]) for key,
                value in invar_cpu.items()}
            true_outvar_cpu = {key: (value + [true_outvar[key].cpu().detach
                ()]) for key, value in true_outvar_cpu.items()}
            pred_outvar_cpu = {key: (value + [pred_outvar[key].cpu().detach
                ()]) for key, value in pred_outvar_cpu.items()}
        invar_cpu = {key: paddle.concat(x=value) for key, value in
            invar_cpu.items()}
        true_outvar_cpu = {key: paddle.concat(x=value) for key, value in
            true_outvar_cpu.items()}
        pred_outvar_cpu = {key: paddle.concat(x=value) for key, value in
            pred_outvar_cpu.items()}
        losses = DeepONet_Data_Validator._l2_relative_error(true_outvar_cpu,
            pred_outvar_cpu)
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}
        named_true_outvar = {('true_' + k): v for k, v in true_outvar.items()}
        named_pred_outvar = {('pred_' + k): v for k, v in pred_outvar.items()}
        if 'np' in save_filetypes:
            np.savez(results_dir + name, {**invar, **named_true_outvar, **
                named_pred_outvar})
        if self.plotter is not None:
            self.plotter._add_figures('Validators', name, results_dir,
                writer, step, self.invar_trunk_plotter, true_outvar,
                pred_outvar)
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar('val/' + name + '/' + k, loss, step,
                    )
            else:
                writer.add_scalar('Validators/' + name + '/' + k, loss,
                    step)
        return losses
