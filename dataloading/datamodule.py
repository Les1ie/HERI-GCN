from datetime import timedelta
from sys import platform
from typing import Optional, List, Union, Any

import torch
from dgl.dataloading import GraphDataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from utils import collate
from .dataset import DiffusionDataset, WeiboTopicDataset


class BaseDataModule(LightningDataModule):

    def __init__(self
                 , data_name: str = 'twitter'
                 , batch_size: int = 4
                 , time_window: Union[int, timedelta] = timedelta(hours=2)
                 , hop=1
                 , raw_dir="data"
                 , sample_rate=1
                 , force_reload: bool = False
                 , num_workers: int = 4
                 , *args, **kwargs):
        super(BaseDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset_name = data_name

        if isinstance(time_window, int):
            time_window = timedelta(hours=time_window)
        self.time_window = time_window
        self.raw_dir = raw_dir
        self.force_reload = force_reload
        self.hop = hop
        self.args = args
        self.sample_rate = sample_rate
        if platform.startswith('win'):
            print('Set num_workers to 0, only support num_workers=0 on Windows platform.')
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        self.kwargs = kwargs
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('MarcoTaskDataModule')
        parser.add_argument('--data_name', type=str, default='twitter',
                            help='Name of dataset to use (default %(default)s).')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='Batch size in training, validation and testing (default %(default)s).')
        parser.add_argument('--hop', type=int, default=1,
                            help='Number of hop in neighbor sampling (default %(default)s).')
        parser.add_argument('--time_window', type=int, default=2,
                            help='Time window of observation (default %(default)s).')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of process threads (default %(default)s).')
        parser.add_argument('--min_cascade_length', type=int, default=20,
                            help='Minimum length of cascades, short cascades will be ignore (default %(default)s).')
        parser.add_argument('--force_reload', action='store_true',
                            help='Reload dataset from raw files.')
        parser.add_argument('--raw_dir', default='data',
                            help='Director of dataset raw files (default ./%(default)s).')
        parser.add_argument('--as_label', default='user', choices=['user', 'repost'],
                            help='The number of user or repost in cascade used as label (default ./%(default)s).')
        parser.add_argument('--sample_rate', default=1, type=float,
                            help='Sample rate from raw data, no greater than 1, (default ./%(default)s).')

        return parent_parser


class MarcoTaskDataModule(BaseDataModule):
    """
    Data module used for macro prediction task.
    """

    def __init__(self
                 , data_name: str = 'twitter'
                 , batch_size: int = 4
                 , time_window: Union[int, timedelta] = timedelta(hours=2)
                 , hop=2
                 , num_workers: int = 4
                 , force_reload: bool = False
                 , *args, **kwargs):
        super(MarcoTaskDataModule, self).__init__(data_name=data_name, batch_size=batch_size, time_window=time_window,
                                                  num_workers=num_workers,
                                                  hop=hop, force_reload=force_reload, *args, **kwargs)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            if not self.train_dataset:
                self.train_dataset = DiffusionDataset(self.dataset_name, 'train', 'macro'
                                                      , time_window=self.time_window
                                                      , force_reload=self.force_reload
                                                      , hop=self.hop
                                                      , *self.args, **self.kwargs)
            if not self.valid_dataset:
                self.valid_dataset = DiffusionDataset(self.dataset_name, 'valid', 'macro'
                                                      , time_window=self.time_window
                                                      , force_reload=self.force_reload
                                                      , hop=self.hop
                                                      , *self.args, **self.kwargs)
        elif stage == 'test' and not self.test_dataset:
            self.test_dataset = DiffusionDataset(self.dataset_name, 'test', 'macro'
                                                 , time_window=self.time_window
                                                 , force_reload=self.force_reload
                                                 , hop=self.hop
                                                 , *self.args, **self.kwargs)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset
                          , collate_fn=collate
                          , batch_size=self.batch_size
                          , num_workers=self.num_workers
                          , shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_dataset
                          , collate_fn=collate
                          , num_workers=self.num_workers
                          , batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset
                          , collate_fn=collate
                          , num_workers=self.num_workers
                          , batch_size=self.batch_size)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        return batch

    @property
    def etypes(self):
        return [
            ('user', 'repost', 'user'),
            ('user', 'follow', 'user')
        ]

    @property
    def hyperparameters(self):
        return {
            'dataset': self.dataset_name,
            'batch_size': self.batch_size,
            'time_window': str(self.time_window),
            'hop': self.hop,
        }

    @property
    def has_processed(self):
        # todo: 完成预处理标记相关的逻辑。
        return False


class WeiboTopicDataModule(BaseDataModule):

    def __init__(self
                 , data_name: str = 'topic'
                 , batch_size: int = 4
                 , time_window: Union[int, timedelta] = timedelta(hours=2)
                 , hop=1
                 , raw_dir="F:\Python-projects\DatasetAnalysis\data\dataset"
                 , force_reload: bool = False
                 , force_rebuild_u2id=True
                 , filter_homo_graph=True
                 , on_multi_files=True
                 , as_label='user'
                 , sample_rate=1
                 , num_workers=4
                 , min_cascade_length: int = 20
                 , time='relative'
                 , train_transforms=None, val_transforms=None, test_transforms=None, dims=None
                 , *args, **kwargs):
        super(WeiboTopicDataModule, self).__init__(data_name=data_name, batch_size=batch_size, time_window=time_window,
                                                   num_workers=num_workers, sample_rate=sample_rate,
                                                   raw_dir=raw_dir, hop=hop, force_reload=force_reload, *args, **kwargs)
        self.on_multi_files = on_multi_files
        self.min_cascade_length = min_cascade_length
        self.filter_homo_graph = filter_homo_graph
        self.force_rebuild_u2id = force_rebuild_u2id
        self.as_label = as_label
        self.time = time
        self._dataset = None

    def process(self, func, save=False, *args, **kwargs):
        self.dataset.process_each(func, save, *args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        if self._dataset and self.train_dataset and self.valid_dataset and self.test_dataset:
            return
        self._dataset = WeiboTopicDataset(name=self.dataset_name,
                                          raw_dir=self.raw_dir,
                                          force_reload=self.force_reload,
                                          time_window=self.time_window,
                                          hop=self.hop,
                                          sample_rate=self.sample_rate,
                                          on_multi_files=self.on_multi_files,
                                          as_label=self.as_label,
                                          min_cascade_length=self.min_cascade_length,
                                          filter_homo_graph=self.filter_homo_graph,
                                          force_rebuild_u2id=self.force_rebuild_u2id,
                                          time=self.time,
                                          )
        datasets = self._dataset.train_valid_test()
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']
        self.test_dataset = datasets['test']

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataloader = GraphDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True
                                     , num_workers=self.num_workers)
        dataloader.dataset = self.train_dataset
        return dataloader

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloader = GraphDataLoader(self.valid_dataset, batch_size=self.batch_size
                                     , num_workers=self.num_workers)
        dataloader.dataset = self.valid_dataset
        return dataloader

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        dataloader = GraphDataLoader(self.test_dataset, batch_size=self.batch_size
                                     , num_workers=self.num_workers)
        dataloader.dataset = self.test_dataset
        return dataloader

    @property
    def dataset(self):
        if self._dataset is None:
            self.setup()
        return self._dataset

    @property
    def has_processed(self):
        return bool(self._dataset) and self._dataset.has_processed

    @property
    def etypes(self):
        return [
            ('user', 'repost', 'user'),
            ('user', 'follow', 'user')
        ]

    @property
    def hyperparameters(self):
        return {
            # 'dataset': 'weib',
            'batch_size': self.batch_size,
            'time_window': str(self.time_window),
            'hop': self.hop,
        }


class MicroTaskDataModule(LightningDataModule):
    """
    Data module used for micro prediction task.
    todo: 完成微观预测数据模块（要求先完成 DiffusionDataset 相关功能）
    """

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass
