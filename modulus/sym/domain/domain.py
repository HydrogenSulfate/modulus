from __future__ import annotations
import paddle
""" Domain
"""
import itertools
import os
from modulus.sym.domain.validator import Validator
from modulus.sym.domain.inferencer import Inferencer
from modulus.sym.domain.monitor import Monitor
from modulus.sym.loss.aggregator import NTK
from modulus.sym.models.arch import FuncArch


class Domain:
    """
    Domain object that contains all needed information about
    constraints, validators, inferencers, and monitors.

    Parameters
    ----------
    name : str
        Unique name for domain.
    encoding : Union[np.ndarray, None]
        Possible encoding vector for domain. Currently not in use.
    """

    def __init__(self, name: str='domain', encoding=None):
        super().__init__()
        self.name = name
        self.encoding = encoding
        self.constraints = {}
        self.validators = {}
        self.inferencers = {}
        self.monitors = {}
        self.ntk = None

    def rec_constraints(self, base_dir: str):
        constraint_data_dir = base_dir + '/constraints/'
        os.makedirs(constraint_data_dir, exist_ok=True)
        for key, constraint in self.constraints.items():
            constraint.save_batch(constraint_data_dir + key)

    def rec_validators(self, base_dir: str, writer: torch.utils.tensorboard
        .SummaryWriter, save_filetypes: str, step: int):
        """Run and save results of validator nodes"""
        validator_data_dir = base_dir + '/validators/'
        os.makedirs(validator_data_dir, exist_ok=True)
        metrics = {}
        for key, validator in self.validators.items():
            valid_losses = validator.save_results(key, validator_data_dir,
                writer, save_filetypes, step)
            if isinstance(valid_losses, dict):
                metrics.update(valid_losses)
        return metrics

    def rec_inferencers(self, base_dir: str, writer: torch.utils.
        tensorboard.SummaryWriter, save_filetypes: str, step: int):
        """Run and save results of inferencer nodes"""
        inferencer_data_dir = base_dir + '/inferencers/'
        os.makedirs(inferencer_data_dir, exist_ok=True)
        for key, inferencer in self.inferencers.items():
            inferencer.save_results(key, inferencer_data_dir, writer,
                save_filetypes, step)

    def rec_stream(self, inferencer, name, base_dir: str, step: int,
        save_results: bool, save_filetypes: str, to_cpu: bool):
        """Run and save results of stream"""
        inferencer_data_dir = base_dir + '/inferencers/'
        if save_results:
            os.makedirs(inferencer_data_dir, exist_ok=True)
        return inferencer.save_stream(name, inferencer_data_dir, None, step,
            save_results, save_filetypes, to_cpu)

    def rec_monitors(self, base_dir: str, writer: torch.utils.tensorboard.
        SummaryWriter, step: int):
        """Run and save results of monitor nodes"""
        monitor_data_dir = base_dir + '/monitors/'
        os.makedirs(monitor_data_dir, exist_ok=True)
        metrics = {}
        for key, monitor in self.monitors.items():
            metrics.update(monitor.save_results(key, writer, step,
                monitor_data_dir))
        return metrics

    def get_num_losses(self):
        return len(set(itertools.chain(*[c.output_names for c in self.
            constraints.values()])))

    def load_data(self, static: bool=False):
        for key, constraint in self.constraints.items():
            if static:
                constraint.load_data_static()
            else:
                constraint.load_data()

    def compute_losses(self, step: int):
        losses = {}
        if self.ntk is None:
            for key, constraint in self.constraints.items():
                paddle.framework.core.nvprof_nvtx_push(
                    f'Constraint Forward: {key}')
                constraint.forward()
                paddle.framework.core.nvprof_nvtx_pop()
            for key, constraint in self.constraints.items():
                for loss_key, value in constraint.loss(step).items():
                    if loss_key not in list(losses.keys()):
                        losses[loss_key] = value
                    else:
                        losses[loss_key] += value
        else:
            losses, self.ntk_weights = self.ntk(self.constraints, self.
                ntk_weights, step)
        return losses

    def get_saveable_models(self):
        models = []
        for c in self.constraints.values():
            if hasattr(c.model, 'module'):
                model = c.model.module
            else:
                model = c.model
            for m in model.evaluation_order:
                if isinstance(m, FuncArch):
                    m = m.arch
                if m not in models and m.saveable:
                    models.append(m)
        models = sorted(models, key=lambda x: x.name)
        assert len(set([m.name for m in models])) == len(models
            ), 'Every model in graph needs a unique name: ' + str([m.name for
            m in models])
        return models

    def create_global_optimizer_model(self):
        models = []
        for c in self.constraints.values():
            if hasattr(c.model, 'module'):
                model = c.model.module
            else:
                model = c.model
            for m in model.optimizer_list:
                if isinstance(m, FuncArch):
                    m = m.arch
                if m not in models:
                    models.append(m)
        models = sorted(models, key=lambda x: x.name)
        assert len(set([m.name for m in models])) == len(models
            ), 'Every model in graph needs a unique name: ' + str([m.name for
            m in models])
        models = paddle.nn.LayerList(sublayers=models)
        return models

    def add_constraint(self, constraint, name: str=None):
        """
        Method to add a constraint to domain.

        Parameters
        ----------
        constraint : Constraint
            Constraint to be added to domain.
        name : str
            Unique name of constraint. If duplicate is
            found then name is iterated to avoid duplication.
        """
        name = Domain._iterate_name(name, 'pointwise_bc', list(self.
            constraints.keys()))
        self.constraints[name] = constraint

    def add_validator(self, validator: Validator, name: str=None):
        """
        Method to add a validator to domain.

        Parameters
        ----------
        validator : Validator
            Validator to be added to domain.
        name : str
            Unique name of validator. If duplicate is
            found then name is iterated to avoid duplication.
        """
        name = Domain._iterate_name(name, 'validator', list(self.validators
            .keys()))
        self.validators[name] = validator

    def add_inferencer(self, inferencer: Inferencer, name: str=None):
        """
        Method to add a inferencer to domain.

        Parameters
        ----------
        inferencer : Inferencer
            Inferencer to be added to domain.
        name : str
            Unique name of inferencer. If duplicate is
            found then name is iterated to avoid duplication.
        """
        name = Domain._iterate_name(name, 'inferencer', list(self.
            inferencers.keys()))
        self.inferencers[name] = inferencer

    def add_monitor(self, monitor: Monitor, name: str=None):
        """
        Method to add a monitor to domain.

        Parameters
        ----------
        monitor : Monitor
            Monitor to be added to domain.
        name : str
            Unique name of monitor. If duplicate is
            found then name is iterated to avoid duplication.
        """
        name = Domain._iterate_name(name, 'monitor', list(self.monitors.keys())
            )
        self.monitors[name] = monitor

    def add_ntk(self, ntk: NTK):
        self.ntk = ntk
        self.ntk_weights = {}

    @staticmethod
    def _iterate_name(input_name, default_name, current_names):
        if input_name is None:
            name = default_name
        else:
            name = input_name
        if name in current_names:
            i = 2
            while True:
                if name + '_' + str(i) not in current_names:
                    name = name + '_' + str(i)
                    break
                i += 1
        return name
