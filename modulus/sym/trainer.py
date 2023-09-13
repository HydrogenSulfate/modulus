import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
""" Modulus Solver
"""
import os
import time
import numpy as np
from termcolor import colored, cprint
from copy import copy
from operator import add
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools
from collections import Counter
from typing import Dict, List, Optional
import logging
from contextlib import ExitStack
from .domain.constraint import Constraint
from .domain import Domain
from .loss.aggregator import Sum
from .utils.training.stop_criterion import StopCriterion
from .constants import TF_SUMMARY, JIT_PYTORCH_VERSION
from .hydra import instantiate_optim, instantiate_sched, instantiate_agg, add_hydra_run_path
from .distributed.manager import DistributedManager
from visualdl import LogWriter

class AdamMixin:
    """Special functions for training using the standard optimizers
    Should be used with ADAM, SGD, RMSProp, etc.
    """

    def adam_compute_gradients(self, aggregator: paddle.nn.Layer,
        global_optimizer_model: paddle.nn.Layer, step: int):
        loss, losses = 0, Counter({})
        for agg_step in range(self.grad_agg_freq):
            with paddle.amp.auto_cast(enable=self.amp, dtype=self.amp_dtype):
                paddle.framework.core.nvprof_nvtx_push('Loss computation')
                losses_minibatch = self.compute_losses(step)
                paddle.framework.core.nvprof_nvtx_pop()
                losses_minibatch = {key: (value / self.grad_agg_freq) for 
                    key, value in losses_minibatch.items()}
                paddle.framework.core.nvprof_nvtx_push('Loss aggregator')
                loss_minibatch = aggregator(losses_minibatch, step)
                paddle.framework.core.nvprof_nvtx_pop()
                loss += loss_minibatch
            paddle.framework.core.nvprof_nvtx_push('Weight gradients')
            self.scaler.scale(loss_minibatch).backward()
            paddle.framework.core.nvprof_nvtx_pop()
            losses.update(losses_minibatch)
        return loss, dict(losses)

    def adam_apply_gradients(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()


class AdaHessianMixin:
    """Special functions for training using the higher-order optimizer AdaHessian"""

    def adahess_compute_gradients(self, aggregator: paddle.nn.Layer,
        global_optimizer_model: paddle.nn.Layer, step: int):
        if self.amp:
            raise NotImplementedError(
                'AMP is not supported for this optimizer.')
        loss, losses = 0, Counter({})
        grads = [paddle.zeros_like(x=parameter) for parameter in list(
            global_optimizer_model.parameters())]
        for agg_step in range(self.grad_agg_freq):
            losses_minibatch = self.compute_losses(step)
            losses_minibatch = {key: (value / self.grad_agg_freq) for key,
                value in losses_minibatch.items()}
            loss_minibatch = aggregator(losses_minibatch, step)
            grads_step = paddle.grad(outputs=loss_minibatch, inputs=list(
                global_optimizer_model.parameters()), create_graph=True)
            grads = list(map(add, grads, grads_step))
            loss += loss_minibatch
            losses.update(losses_minibatch)
        for grad, param in zip(grads, global_optimizer_model.parameters()):
            param.grad = grad
        return loss, dict(losses)

    def adahess_apply_gradients(self):
        self.adam_apply_gradients()


class BFGSMixin:
    """Special functions for training using BFGS optimizer"""

    def bfgs_compute_gradients(self, aggregator: paddle.nn.Layer,
        global_optimizer_model: paddle.nn.Layer, step: int):
        if self.amp:
            raise NotImplementedError(
                'AMP is not supported for this optimizer.')
        if self.max_steps != 0:
            self.log.warning('lbfgs optimizer selected. Setting max_steps to 0'
                )
            self.max_steps = 0
        if self.grad_agg_freq != 1:
            self.log.warning(
                'lbfgs optimizer selected. Setting grad_agg_freq to 1')
            self.grad_agg_freq = 1
        losses = self.compute_losses(step)
        loss = aggregator(losses, step)
        self.bfgs_step = step
        self.bfgs_aggregator = aggregator
        for param in global_optimizer_model.parameters():
            param.grad = None
        return loss, losses

    def bfgs_closure_func(self):
        """Class Method: *.clear_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        self.optimizer.clear_grad()
        loss = 0
        losses = self.compute_losses(self.bfgs_step)
        loss = self.bfgs_aggregator(losses, self.bfgs_step)
        loss.backward()
        self.bfgs_optim_steps += 1
        return loss

    def bfgs_apply_gradients(self):
        assert not self.bfgs_aggregator is None, 'Call bfgs_compute_gradients prior to this!'
        assert not self.bfgs_step is None, 'Call bfgs_compute_gradients prior to this!'
        self.bfgs_optim_steps = 0
        self.log.info(
            f'[step: {self.bfgs_step:10d}] lbfgs optimization in running')
        self.optimizer.step(self.bfgs_closure_func)
        self.log.info(
            f'lbfgs optimization completed after {self.bfgs_optim_steps} steps'
            )


class Trainer(AdamMixin, AdaHessianMixin, BFGSMixin):
    """Base class for optimizing networks on losses/constraints"""

    def __init__(self, cfg: DictConfig):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self._network_dir = self.cfg.network_dir
        self._initialization_network_dir = self.cfg.initialization_network_dir
        self.max_steps = self.cfg.training.max_steps
        self.grad_agg_freq = self.cfg.training.grad_agg_freq
        self.save_network_freq = self.cfg.training.save_network_freq
        self.print_stats_freq = self.cfg.training.print_stats_freq
        self.summary_freq = self.cfg.training.summary_freq
        self.amp = self.cfg.training.amp
        self.stop_criterion_metric = self.cfg.stop_criterion.metric
        self.stop_criterion_min_delta = self.cfg.stop_criterion.min_delta
        self.stop_criterion_patience = self.cfg.stop_criterion.patience
        """Class Attribute: torch.distributions.Distribution.mode, can not convert, please check whether it is torch.Tensor.*/torch.autograd.function.FunctionCtx.*/torch.distributions.Distribution.* and convert manually"""
        self.stop_criterion_mode = self.cfg.stop_criterion.mode
        self.stop_criterion_freq = self.cfg.stop_criterion.freq
        self.stop_criterion_strict = self.cfg.stop_criterion.strict
        self.save_filetypes = self.cfg.save_filetypes
        self.summary_histograms = self.cfg.summary_histograms
        self.apply_gradients = self._apply_gradients
        self.compute_gradients = self._compute_gradients
        self.log = logging.getLogger(__name__)
        self.manager = DistributedManager()
        self.place = self.manager.place
        self.device_amp = 'cuda' if self.manager.cuda else 'cpu'
        if (self.cfg.training.amp_dtype == 'bfloat16' or self.device_amp ==
            'cpu'):
            self.amp_dtype = 'bfloat16'
            if self.device_amp == 'cpu' and self.amp:
                self.log.warning(
                    'Switching amp_dtype to bfloat16, AutocastCPU only supports bfloat16'
                    )
        else:
            self.amp_dtype = 'float16'

    def compute_losses(self, step: int):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def _compute_gradients(self):
        raise NotImplementedError(
            'Config should set the compute_gradients function')

    def _apply_gradients(self):
        raise NotImplementedError(
            'Config should set the apply_gradients function')

    def get_saveable_models(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def create_global_optimizer_model(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def load_network(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def save_checkpoint(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def record_constraints(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def record_validators(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    @property
    def has_validators(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def record_inferencers(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    @property
    def has_inferencers(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def record_monitors(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    @property
    def has_monitors(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def get_num_losses(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def _record_constraints(self):
        data_parallel_rank = self.manager.group_rank('data_parallel'
            ) if self.manager.distributed else 0
        if data_parallel_rank == 0:
            rec_inferencer_start = time.time()
            self.record_constraints()
            self.log.debug(
                f'{self.step_str} saved constraint results to {self.network_dir}'
                )
            self.log.info(
                f'{self.step_str} record constraint batch time: {time.time() - rec_inferencer_start:10.3e}s'
                )

    def _record_validators(self, step):
        data_parallel_rank = self.manager.group_rank('data_parallel'
            ) if self.manager.distributed else 0
        if data_parallel_rank == 0:
            rec_validation_start = time.time()
            self.validator_outvar = self.record_validators(step)
            self.log.debug(
                f'{self.step_str} saved validator results to {self.network_dir}'
                )
            self.log.info(
                f'{self.step_str} record validators time: {time.time() - rec_validation_start:10.3e}s'
                )

    def _record_inferencers(self, step):
        data_parallel_rank = self.manager.group_rank('data_parallel'
            ) if self.manager.distributed else 0
        if data_parallel_rank == 0:
            rec_inferencer_start = time.time()
            self.record_inferencers(step)
            self.log.debug(
                f'{self.step_str} saved inferencer results to {self.network_dir}'
                )
            self.log.info(
                f'{self.step_str} record inferencers time: {time.time() - rec_inferencer_start:10.3e}s'
                )

    def _record_monitors(self, step):
        data_parallel_rank = self.manager.group_rank('data_parallel'
            ) if self.manager.distributed else 0
        if data_parallel_rank == 0:
            rec_monitor_start = time.time()
            self.monitor_outvar = self.record_monitors(step)
            self.log.debug(
                f'{self.step_str} saved monitor results to {self.network_dir}')
            if self.summary_histograms:
                for name, parameter in self.global_optimizer_model.named_parameters(
                    ):
                    name = name.split('.')
                    name = '.'.join(name[:-1]) + '/' + '.'.join(name[-1:])
                    self.writer.add_histogram(name, parameter.detach().
                        flatten(), step)
                    if parameter.grad is not None:
                        self.writer.add_histogram(name + '_gradient',
                            parameter.grad.detach().flatten(), step)
            self.log.info(
                f'{self.step_str} record monitor time: {time.time() - rec_monitor_start:10.3e}s'
                )

    def _check_stopping_criterion(self, loss, losses, step):
        if self.manager.rank == 0:
            if self.stop_criterion_metric is None:
                return False
            elif step % self.stop_criterion_freq == 0:
                criterion_metric_dict = {'loss': {'loss': loss.cpu().detach
                    ().numpy()}}
                criterion_metric_dict['loss'].update({key: val.cpu().detach
                    ().numpy() for key, val in losses.items()})
                if self.has_monitors:
                    criterion_metric_dict.update({'monitor': {key: val.cpu(
                        ).detach().numpy() for key, val in self.
                        monitor_outvar.items()}})
                if self.has_validators:
                    criterion_metric_dict.update({'validation': {key: val.
                        cpu().detach().numpy() for key, val in self.
                        validator_outvar.items()}})
                stop_training = self.stop_criterion.evaluate(
                    criterion_metric_dict)
                return stop_training
            else:
                return False

    def _train_loop(self, sigterm_handler=None):
        if self.manager.rank == 0:
            os.makedirs(self.network_dir, exist_ok=True)
        self.saveable_models = self.get_saveable_models()
        self.global_optimizer_model = self.create_global_optimizer_model()
        self.compute_gradients = getattr(self, self.cfg.optimizer._params_.
            compute_gradients)
        self.apply_gradients = getattr(self, self.cfg.optimizer._params_.
            apply_gradients)
        self.optimizer = instantiate_optim(self.cfg, model=self.
            global_optimizer_model)
        self.scheduler = instantiate_sched(self.cfg, optimizer=self.optimizer)
        self.aggregator = instantiate_agg(self.cfg, model=self.
            global_optimizer_model.parameters(), num_losses=self.
            get_num_losses())
        if self.cfg.jit:
            if not paddle.__version__ == JIT_PYTORCH_VERSION:
                self.log.warn(
                    f'Installed PyTorch version {paddle.__version__} is not TorchScript'
                     +
                    f' supported in Modulus. Version {JIT_PYTORCH_VERSION} is officially supported.'
                    )
            self.aggregator = torch.jit.script(self.aggregator)
            if self.amp:
                torch._C._jit_set_autocast_mode(True)
        if len(list(self.aggregator.parameters())) > 0:
            self.log.debug(
                'Adding loss aggregator param group. LBFGS will not work!')
            """Class Method: *.add_param_group, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
            self.optimizer.add_param_group({'params': list(self.aggregator.
                parameters())})
        enable_scaler = self.amp and self.amp_dtype == 'float16'
        self.scaler = paddle.amp.GradScaler(enable=enable_scaler,
            incr_every_n_steps=2000, init_loss_scaling=65536.0)
        if self.stop_criterion_metric is not None:
            self.stop_criterion = StopCriterion(self.stop_criterion_metric,
                self.stop_criterion_min_delta, self.stop_criterion_patience,
                self.stop_criterion_mode, self.stop_criterion_freq, self.
                stop_criterion_strict, self.cfg.training.rec_monitor_freq,
                self.cfg.training.rec_validation_freq)
        self.initial_step = self.load_network()
        # self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.
        #     network_dir, purge_step=self.summary_freq + 1)
        self.writer = LogWriter(logdir=self.network_dir)
        self.summary_histograms = self.cfg['summary_histograms']
        if self.manager.rank == 0:
            self.writer.add_text('config',
                f'<pre>{str(OmegaConf.to_yaml(self.cfg))}</pre>')
        try:
            self.profile = self.cfg.profiler.profile
            self.profiler_start_step = self.cfg.profiler.start_step
            self.profiler_end_step = self.cfg.profiler.end_step
            if self.profiler_end_step < self.profiler_start_step:
                self.profile = False
        except:
            self.profile = False
            self.profiler_start_step = -1
            self.profiler_end_step = -1
        if self.manager.distributed:
            torch.distributed.barrier()
        barrier_flag = False
        if self.manager.cuda:
            start_event = paddle.device.cuda.Event(enable_timing=True)
            # end_event = paddle.device.cuda.Event(enable_timing=True)
            start_event.record()
            t = time.time()
        else:
            t = time.time()
        if sigterm_handler is None:
            self.sigterm_handler = lambda : False
        else:
            self.sigterm_handler = sigterm_handler
        with ExitStack() as stack:
            if self.profile:
                self.log.warning('Running in profiling mode')
                stack.enter_context(torch.autograd.profiler.emit_nvtx())
            for step in range(self.initial_step, self.max_steps + 1):
                if self.sigterm_handler():
                    if self.manager.rank == 0:
                        self.log.info(
                            f'Training terminated by the user at iteration {step}'
                            )
                    break
                if self.profile and step == self.profiler_start_step:
                    self.log.info('Starting profiler at step {}'.format(step))
                    paddle.profiler.start()
                if self.profile and step == self.profiler_end_step:
                    self.log.info('Stopping profiler at step {}'.format(step))
                    paddle.profiler.stop()
                paddle.framework.core.nvprof_nvtx_push('Training iteration')
                if self.cfg.cuda_graphs:
                    self.load_data(static=True)
                    loss, losses = self._cuda_graph_training_step(step)
                else:
                    self.load_data()
                    """Class Method: *.clear_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
                    self.optimizer.clear_grad()
                    loss, losses = self.compute_gradients(self.aggregator,
                        self.global_optimizer_model, step)
                    self.apply_gradients()
                    self.scheduler.step()
                if paddle.isnan(x=loss):
                    self.log.error('loss went to Nans')
                    break
                self.step_str = f'[step: {step:10d}]'
                if step % self.summary_freq == 0:
                    if self.manager.rank == 0:
                        for key, value in losses.items():
                            if TF_SUMMARY:
                                self.writer.add_scalar('Train_/loss_L2' +
                                    str(key), value, step)
                            else:
                                self.writer.add_scalar('Train/loss_' + str(
                                    key), value, step)
                        if TF_SUMMARY:
                            self.writer.add_scalar('Optimzer/loss', loss,
                                step)
                            self.writer.add_scalar('learning_rate/lr', self
                                .optimizer.get_lr(), step)
                        else:
                            self.writer.add_scalar('Train/loss_aggregated',
                                loss, step)
                            self.writer.add_scalar('Train/learning_rate',
                                self.optimizer.get_lr(), step)
                    if self.manager.distributed:
                        barrier_flag = True
                if step % self.cfg.training.rec_constraint_freq == 0:
                    barrier_flag = True
                    self._record_constraints()
                if (step % self.cfg.training.rec_validation_freq == 0 and
                    self.has_validators):
                    barrier_flag = True
                    self._record_validators(step)
                if (step % self.cfg.training.rec_inference_freq == 0 and
                    self.has_inferencers):
                    barrier_flag = True
                    self._record_inferencers(step)
                if (step % self.cfg.training.rec_monitor_freq == 0 and self
                    .has_monitors):
                    barrier_flag = True
                    self._record_monitors(step)
                if step % self.save_network_freq == 0:
                    data_parallel_rank = self.manager.group_rank(
                        'data_parallel') if self.manager.distributed else 0
                    if data_parallel_rank == 0:
                        self.save_checkpoint(step)
                        self.log.info(
                            f'{self.step_str} saved checkpoint to {add_hydra_run_path(self.network_dir)}'
                            )
                    if self.manager.distributed:
                        barrier_flag = True
                if self.manager.distributed and barrier_flag:
                    paddle.distributed.barrier()
                    barrier_flag = False
                if step % self.print_stats_freq == 0:
                    if self.manager.cuda:
                        # end_event.record()
                        # end_event.synchronize()
                        # elapsed_time = start_event.elapsed_time(end_event)
                        t_end = time.time()
                        elapsed_time = (t_end - t) * 1000.0
                    else:
                        t_end = time.time()
                        elapsed_time = (t_end - t) * 1000.0
                    if self.manager.distributed:
                        paddle.distributed.reduce(loss, 0, op=paddle.distributed.ReduceOp.AVG)
                        elapsed_time = paddle.to_tensor(data=elapsed_time).to(
                            self.place)
                        paddle.distributed.reduce(elapsed_time, 0, op=paddle.distributed.ReduceOp.AVG)
                        elapsed_time = elapsed_time.cpu().numpy()[()]
                    print_statement = (
                        f'{self.step_str} loss: {loss.cpu().detach().numpy().item():10.3e}'
                        )
                    if step >= self.initial_step + self.print_stats_freq:
                        print_statement += (
                            f', time/iteration: {elapsed_time / self.print_stats_freq:10.3e} ms'
                            )
                    if self.manager.rank == 0:
                        self.log.info(print_statement)
                    if self.manager.cuda:
                        start_event.record()
                    else:
                        t = time.time()
                stop_training = self._check_stopping_criterion(loss, losses,
                    step)
                if stop_training:
                    if self.manager.rank == 0:
                        self.log.info(
                            f'{self.step_str} stopping criterion is met, finished training!'
                            )
                    break
                if step >= self.max_steps:
                    if self.manager.rank == 0:
                        self.log.info(
                            f'{self.step_str} reached maximum training steps, finished training!'
                            )
                    break
                paddle.framework.core.nvprof_nvtx_pop()

    def _cuda_graph_training_step(self, step: int):
        raise NotImplementedError(
            f"Paddle do not support CUDA graph now."
        )
#         if step - self.initial_step < self.cfg.cuda_graph_warmup:
#             if step - self.initial_step == 0:
#                 self.warmup_stream = paddle.device.cuda.Stream()
#             self.warmup_stream.wait_stream(paddle.device.cuda.current_stream())
#             with paddle.cuda.stream(self.warmup_stream):
#                 """Class Method: *.clear_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
#                 self.global_optimizer_model.clear_grad(set_to_none=True)
#                 self.loss_static, self.losses_static = self.compute_gradients(
#                     self.aggregator, self.global_optimizer_model, step)
#             paddle.device.cuda.current_stream().wait_stream(self.warmup_stream)
#             self.apply_gradients()
#             self.scheduler.step()
#         elif step - self.initial_step == self.cfg.cuda_graph_warmup:
#             paddle.device.cuda.synchronize()
#             if self.manager.distributed:
#                 torch.distributed.barrier()
#             if self.cfg.cuda_graph_warmup < 11:
#                 self.log.warn(
#                     f'Graph warm up length ({self.cfg.cuda_graph_warmup}) should be more than 11 steps, higher suggested'
#                     )
#             self.log.info(
#                 'Attempting cuda graph building, this may take a bit...')
#             self.g = torch.cuda.CUDAGraph()
#             """Class Method: *.clear_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>            self.global_optimizer_model.clear_grad(set_to_none=True)
#             delay = os.environ.get('MODULUS_CUDA_GRAPH_CAPTURE_DELAY', '10')
#             time.sleep(int(delay))
# >>>            with torch.cuda.graph(self.g):
#                 self.loss_static, self.losses_static = self.compute_gradients(
#                     self.aggregator, self.global_optimizer_model, step)
#             self.apply_gradients()
#             self.scheduler.step()
#         else:
#             self.g.replay()
#             self.apply_gradients()
#             self.scheduler.step()
#         return self.loss_static, self.losses_static

    def _eval(self):
        if not os.path.exists(self.network_dir):
            raise RuntimeError('Network checkpoint is required for eval mode.')
        self.saveable_models = self.get_saveable_models()
        if self.place is None:
            self.place = self.manager.place
        self.step = self.load_step()
        self.step = self.load_model()
        self.step_str = f'[step: {self.step:10d}]'
        # self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.
        #     network_dir, purge_step=self.summary_freq + 1)
        self.writer = LogWriter(logdir=self.network_dir)
        self.summary_histograms = self.cfg['summary_histograms']
        if self.manager.cuda:
            paddle.device.cuda.synchronize()
        if self.has_validators:
            self._record_validators(self.step)
        if self.has_inferencers:
            self._record_inferencers(self.step)
        if self.has_monitors:
            self._record_monitors(self.step)

    def _stream(self):
        if not os.path.exists(self.network_dir):
            raise RuntimeError(
                'Network checkpoint is required for stream mode.')
        self.saveable_models = self.get_saveable_models()
        if self.place is None:
            self.place = self.manager.place
        self.step = self.load_step()
        self.step = self.load_model()
        self.step_str = f'[step: {self.step:10d}]'
        if self.manager.cuda:
            paddle.device.cuda.synchronize()
        return self.record_stream

    @staticmethod
    def _load_network(initialization_network_dir: str, network_dir: str,
        models: List[paddle.nn.Layer], optimizer: paddle.optimizer.
        Optimizer, aggregator: paddle.nn.Layer, scheduler: paddle.optimizer.
        lr.LRScheduler, scaler: paddle.amp.GradScaler, log:
        logging.Logger, manager: DistributedManager, device: Optional[str]=None
        ):
        if device is None:
            device = manager.place
        step = Trainer._load_optimizer(network_dir, optimizer, aggregator,
            scheduler, scaler, log, device)
        step = Trainer._load_model(initialization_network_dir, network_dir,
            models, step, log, device)
        return step

    @staticmethod
    def _load_optimizer(network_dir: str, optimizer: paddle.optimizer.
        Optimizer, aggregator: paddle.nn.Layer, scheduler: paddle.optimizer.
        lr.LRScheduler, scaler: paddle.amp.GradScaler, log:
        logging.Logger, device: str):
        manager = DistributedManager()
        model_parallel_rank = manager.group_rank('model_parallel'
            ) if manager.distributed else 0
        optimizer_checkpoint_file = (network_dir +
            f'/optim_checkpoint.{model_parallel_rank}.pth')
        log.info('attempting to restore from: ' + add_hydra_run_path(
            network_dir))
        if os.path.exists(optimizer_checkpoint_file):
            try:
                checkpoint = paddle.load(path=optimizer_checkpoint_file)
                optimizer.set_state_dict(state_dict=checkpoint[
                    'optimizer_state_dict'])
                aggregator.set_state_dict(state_dict=checkpoint[
                    'aggregator_state_dict'])
                scheduler.set_state_dict(state_dict=checkpoint[
                    'scheduler_state_dict'])
                scaler.set_state_dict(state_dict=checkpoint[
                    'scaler_state_dict'])
                step = checkpoint['step']
                success = colored('Success loading optimizer: ', 'green')
                log.info(success + add_hydra_run_path(
                    optimizer_checkpoint_file))
            except:
                fail = colored('Fail loading optimizer: ', 'red')
                step = 0
                log.info(fail + add_hydra_run_path(network_dir +
                    '/optim_checkpoint.pth'))
        else:
            log.warning('optimizer checkpoint not found')
            step = 0
        return step

    @staticmethod
    def _load_model(initialization_network_dir: str, network_dir: str,
        models: List[paddle.nn.Layer], step: int, log: logging.Logger,
        device: str):
        manager = DistributedManager()
        model_parallel_rank = manager.group_rank('model_parallel'
            ) if manager.distributed else 0
        if initialization_network_dir != '':
            for i_dir in initialization_network_dir.split(','):
                if os.path.exists(i_dir):
                    log.info('attempting to initialize network from ' + i_dir)
                    for model in models:
                        if os.path.exists(i_dir + '/' + model.
                            checkpoint_filename):
                            try:
                                model.load(i_dir, map_location=device)
                                success = colored('Success loading model: ',
                                    'green')
                                log.info(success + i_dir + '/' + model.
                                    checkpoint_filename)
                            except:
                                fail = colored('Fail loading model: ', 'red')
                                step = 0
                                log.error(fail + i_dir + '/' + model.
                                    checkpoint_filename)
                        else:
                            log.warning('model ' + model.
                                checkpoint_filename +
                                ' not found for initialization')
        for model in models:
            if os.path.exists(network_dir + '/' + model.checkpoint_filename):
                try:
                    model.load(network_dir, map_location=device)
                    success = colored('Success loading model: ', 'green')
                    log.info(success + add_hydra_run_path(network_dir + '/' +
                        model.checkpoint_filename))
                except:
                    fail = colored('Fail loading model: ', 'red')
                    log.info(fail + add_hydra_run_path(network_dir + '/' +
                        model.checkpoint_filename))
            else:
                log.warning('model ' + model.checkpoint_filename + ' not found'
                    )
                step = 0
        return step

    @staticmethod
    def _load_step(network_dir: str, device: Optional[str]=None):
        manager = DistributedManager()
        model_parallel_rank = manager.group_rank('model_parallel'
            ) if manager.distributed else 0
        if os.path.exists(network_dir +
            f'/optim_checkpoint.{model_parallel_rank}.pth'):
            try:
                checkpoint = paddle.load(path=network_dir +
                    f'/optim_checkpoint.{model_parallel_rank}.pth')
                step = checkpoint['step']
            except:
                step = 0
        else:
            step = 0
        return step

    @staticmethod
    def _save_checkpoint(network_dir: str, models: List[paddle.nn.Layer],
        optimizer: paddle.optimizer.Optimizer, aggregator: paddle.nn.Layer,
        scheduler: paddle.optimizer.lr.LRScheduler, scaler: paddle.
        amp.GradScaler, step: int):
        manager = DistributedManager()
        model_parallel_rank = manager.group_rank('model_parallel'
            ) if manager.distributed else 0
        for model in models:
            model.save(network_dir)
        paddle.save(obj={'step': step, 'optimizer_state_dict': optimizer.
            state_dict(), 'aggregator_state_dict': aggregator.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()}, path=network_dir +
            f'/optim_checkpoint.{model_parallel_rank}.pth')
