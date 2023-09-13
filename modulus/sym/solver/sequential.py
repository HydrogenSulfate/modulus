import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import os
import numpy as np
from typing import List, Union, Tuple, Callable
from omegaconf import DictConfig
import warnings
from modulus.sym.distributed.manager import DistributedManager
from modulus.sym.trainer import Trainer
from modulus.sym.domain import Domain
from modulus.sym.loss.aggregator import NTK
from .solver import Solver


class SequentialSolver(Solver):
    """
    Solver class for solving a sequence of domains.
    This solver can be used to set up iterative methods
    like the hFTB conjugate heat transfer method or
    the moving time window method for transient problems.

    Parameters
    ----------
    cfg : DictConfig
        Hydra dictionary of configs.
    domains : List[Tuple[int, Domain]]
        List of Domains to sequentially solve.
        Each domain is given as a tuple where the first
        element is an int for how many times to solve
        the domain and the second element is the domain.
        For example, `domains=[(1, domain_a), (4, domain_b)]`
        would solve `domain_a` once and then solve `domain_b`
        4 times in a row.
    custom_update_operation : Union[Callable, None] = None
        A callable function to update any weights in models.
        This function will be called at the end of every
        iteration.
    """

    def __init__(self, cfg: DictConfig, domains: List[Tuple[int, Domain]],
        custom_update_operation: Union[Callable, None]=None):
        assert len(set([d.name for _, d in domains])) == len(domains
            ), 'domains need to have unique names, ' + str([d.name for _, d in
            domains])
        assert not cfg.training.ntk.use_ntk, 'ntk is not supported with SequentialSolver'
        self.domains = domains
        self.custom_update_operation = custom_update_operation
        Trainer.__init__(self, cfg)
        self.load_iteration_step()

    def load_iteration_step(self):
        try:
            iteration_step_file = open(self._network_dir +
                '/current_step.txt', 'r')
            contents = iteration_step_file.readlines()[0]
            domain_index = int(contents.split(' ')[0])
            iteration_index = int(contents.split(' ')[1])
        except:
            domain_index = 0
            iteration_index = 0
        self.domain_index = domain_index
        self.iteration_index = iteration_index

    def save_iteration_step(self):
        iteration_step_file = open(self._network_dir + '/current_step.txt', 'w'
            )
        iteration_step_file.write(str(self.domain_index) + ' ' + str(self.
            iteration_index))

    @property
    def domain(self):
        return self.domains[self.domain_index][1]

    @property
    def network_dir(self):
        dir_name = self._network_dir + '/' + self.domain.name
        if self.domains[self.domain_index][0] > 1:
            dir_name += '_' + str(self.iteration_index).zfill(4)
        return dir_name

    def solve(self, sigterm_handler=None):
        if self.cfg.run_mode == 'train':
            if DistributedManager().rank == 0:
                os.makedirs(self.network_dir, exist_ok=True)
            for domain_index in range(self.domain_index, len(self.domains)):
                for iteration_index in range(self.iteration_index, self.
                    domains[domain_index][0]):
                    self.domain_index = domain_index
                    self.iteration_index = iteration_index
                    self.save_iteration_step()
                    self.log.info('Solving for Domain ' + str(self.domain.
                        name) + ', iteration ' + str(self.iteration_index))
                    self._train_loop(sigterm_handler)
                    if self.custom_update_operation is not None:
                        self.custom_update_operation()
        elif self.cfg.run_mode == 'eval':
            raise NotImplementedError(
                'eval mode not implemented for sequential training')
        else:
            raise RuntimeError('Invalid run mode')
