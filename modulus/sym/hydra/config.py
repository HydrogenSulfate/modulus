import paddle
"""
Modulus main config
"""
from platform import architecture
import logging
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir, HydraConf
from omegaconf import MISSING, SI
from typing import List, Any
from modulus.sym.constants import JIT_PYTORCH_VERSION
from packaging import version
from .loss import LossConf
from .optimizer import OptimizerConf
from .pde import PDEConf
from .scheduler import SchedulerConf
from .training import TrainingConf, StopCriterionConf
from .profiler import ProfilerConf
from .hydra import default_hydra
logger = logging.getLogger(__name__)


@dataclass
class ModulusConfig:
    network_dir: str = '.'
    initialization_network_dir: str = ''
    save_filetypes: str = 'vtk'
    summary_histograms: bool = False
    jit: bool = version.parse(paddle.__version__) >= version.parse(
        JIT_PYTORCH_VERSION)
    jit_use_nvfuser: bool = True
    jit_arch_mode: str = 'only_activation'
    jit_autograd_nodes: bool = False
    cuda_graphs: bool = True
    cuda_graph_warmup: int = 20
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False
    device: str = ''
    debug: bool = False
    run_mode: str = 'train'
    arch: Any = MISSING
    models: Any = MISSING
    training: TrainingConf = MISSING
    stop_criterion: StopCriterionConf = MISSING
    loss: LossConf = MISSING
    optimizer: OptimizerConf = MISSING
    scheduler: SchedulerConf = MISSING
    batch_size: Any = MISSING
    profiler: ProfilerConf = MISSING
    hydra: Any = field(default_factory=lambda : default_hydra)
    custom: Any = MISSING


default_defaults = [{'training': 'default_training'}, {'graph': 'default'},
    {'stop_criterion': 'default_stop_criterion'}, {'profiler': 'nvtx'}, {
    'override hydra/job_logging': 'info_logging'}, {
    'override hydra/launcher': 'basic'}, {'override hydra/help':
    'modulus_help'}, {'override hydra/callbacks': 'default_callback'}]


@dataclass
class DefaultModulusConfig(ModulusConfig):
    defaults: List[Any] = field(default_factory=lambda : default_defaults)


debug_defaults = [{'training': 'default_training'}, {'graph': 'default'}, {
    'stop_criterion': 'default_stop_criterion'}, {'profiler': 'nvtx'}, {
    'override hydra/job_logging': 'debug_logging'}, {'override hydra/help':
    'modulus_help'}, {'override hydra/callbacks': 'default_callback'}]


@dataclass
class DebugModulusConfig(ModulusConfig):
    defaults: List[Any] = field(default_factory=lambda : debug_defaults)
    debug: bool = True


experimental_defaults = [{'training': 'default_training'}, {'graph':
    'default'}, {'stop_criterion': 'default_stop_criterion'}, {'profiler':
    'nvtx'}, {'override hydra/job_logging': 'info_logging'}, {
    'override hydra/launcher': 'basic'}, {'override hydra/help':
    'modulus_help'}, {'override hydra/callbacks': 'default_callback'}]


@dataclass
class ExperimentalModulusConfig(ModulusConfig):
    defaults: List[Any] = field(default_factory=lambda : experimental_defaults)
    pde: PDEConf = MISSING


def register_modulus_configs() ->None:
    if not paddle.__version__ == JIT_PYTORCH_VERSION:
        logger.warn(
            f'TorchScript default is being turned off due to PyTorch version mismatch.'
            )
    cs = ConfigStore.instance()
    cs.store(name='modulus_default', node=DefaultModulusConfig)
    cs.store(name='modulus_debug', node=DebugModulusConfig)
    cs.store(name='modulus_experimental', node=ExperimentalModulusConfig)
