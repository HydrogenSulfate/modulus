import sys
sys.path.append(
    '/workspace/hesensen/paper_reprod/PaConvert/paddle_project_hss/utils')
import paddle_aux
import paddle
from typing import Union, List
import logging
from typing import Union, List
from modulus.sym.node import Node
from modulus.sym.constants import tf_dt
from modulus.sym.distributed.manager import DistributedManager
from modulus.sym.dataset import Dataset, IterableDataset
from modulus.sym.loss import Loss
from modulus.sym.graph import Graph
from modulus.sym.key import Key
logger = logging.getLogger(__name__)
Tensor = paddle.Tensor


class Constraint:
    """Base class for constraints"""

    def __init__(self, nodes: List[Node], dataset: Union[Dataset,
        IterableDataset], loss: Loss, batch_size: int, shuffle: bool,
        drop_last: bool, num_workers: int):
        self.manager = DistributedManager()
        self.place = self.manager.place
        if not drop_last and self.manager.cuda_graphs:
            logger.info('drop_last must be true when using cuda graphs')
            drop_last = True
        self.dataset = dataset
        self.dataloader = iter(Constraint.get_dataloader(dataset=self.
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=
            drop_last, num_workers=num_workers))
        self.model = Graph(nodes, Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys))
        # self.model.to(self.place)
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
            # paddle.device.cuda.current_stream().wait_stream(s)
        self._input_names = Key.convert_list(dataset.invar_keys)
        self._output_names = Key.convert_list(dataset.outvar_keys)
        self._input_vars = None
        self._target_vars = None
        self._lambda_weighting = None
        self._loss = loss

    @property
    def input_names(self) ->List[Key]:
        return self._input_names

    @property
    def output_names(self) ->List[Key]:
        return self._output_names

    def load_data(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def load_data_static(self):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def loss(self, step: int):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    def save_batch(self, filename: str):
        raise NotImplementedError(
            'Subclass of Constraint needs to implement this')

    @staticmethod
    def _set_device(tensor_dict, device=None, requires_grad=False):
        tensor_dict = {key: paddle.to_tensor(data=value, place=device).
            astype(tf_dt) for key, value in tensor_dict.items()}
        if requires_grad:
            for k, v in tensor_dict.items():
                v.stop_gradient = not requires_grad
        return tensor_dict

    @staticmethod
    def get_dataloader(dataset: Union[Dataset, IterableDataset], batch_size:
        int, shuffle: bool, drop_last: bool, num_workers: int, distributed:
        bool=None, infinite: bool=True):
        """Return an appropriate dataloader given a dataset"""
        assert isinstance(dataset, Dataset) or isinstance(dataset,
            IterableDataset
            ), 'error, dataset must be a subclass of Dataset or IterableDataset'
        manager = DistributedManager()
        persistent_workers = True if num_workers > 0 else False
        if isinstance(dataset, Dataset):
            assert batch_size is not None, 'error, batch_size must be specified'
            assert shuffle is not None, 'error, shuffle must be specified'
            assert drop_last is not None, 'error, drop_last must be specified'
            if distributed is not False and manager.distributed:
                sampler = paddle.io.DistributedBatchSampler(dataset=dataset,
                    num_replicas=manager.group_size('data_parallel'), rank=
                    manager.group_rank('data_parallel'), shuffle=shuffle,
                    drop_last=drop_last, batch_size=1)
            elif shuffle:
                sampler = paddle.io.RandomSampler(data_source=dataset)
            else:
                sampler = paddle.io.SequenceSampler(data_source=dataset)
            batch_sampler = paddle.io.BatchSampler(sampler=sampler,
                batch_size=batch_size, drop_last=drop_last)
            if dataset.auto_collation:
                dataloader = paddle.io.DataLoader(dataset,
                    batch_sampler=paddle.io.BatchSampler(sampler=sampler, shuffle=False, batch_size=batch_size),
                    num_workers=num_workers, worker_init_fn=dataset.
                    worker_init_fn, persistent_workers=persistent_workers)
            else:
                dataloader = paddle.io.DataLoader(dataset,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers, worker_init_fn=dataset.
                    worker_init_fn, persistent_workers=persistent_workers)
        elif isinstance(dataset, IterableDataset):
            dataloader = paddle.io.DataLoader(dataset, batch_size=
                None,  num_workers=num_workers,
                worker_init_fn=dataset.worker_init_fn, persistent_workers=
                persistent_workers)
        if infinite:
            dataloader = InfiniteDataLoader(dataloader)
        if num_workers == 0:
            dataset.worker_init_fn(0)
        return dataloader


class InfiniteDataLoader:
    """An infinite dataloader, for use with map-style datasets to avoid StopIteration after each epoch"""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        while True:
            dataloader = iter(self.dataloader)
            for batch in dataloader:
                yield batch
            self.epoch += 1
