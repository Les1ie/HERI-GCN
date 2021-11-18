import os
from concurrent import futures
from datetime import timedelta
from typing import List, Dict

from GAS.dataloading.dataset import WeiboTopicDataset


class DataLoadingTask(Dict):
    def __init__(self, dataset_cls: type, *args, **kwargs):
        self.dataset_cls = dataset_cls
        self.init_args = list(args)
        self.init_kwargs = kwargs
        self._dataset = None

    # @staticmethod
    def load(self):
        print(f'\nLoading {self.__repr__()}.')
        dataset = self.dataset_cls(*self.init_args, **self.init_kwargs)
        self._dataset = dataset
        return self._dataset

    @property
    def dataset(self):
        if self._dataset is None:
            self.load()
        return self._dataset

    def __repr__(self):
        args = [f'{k}={v}' for k, v in self.init_kwargs.items()]
        params = ', '.join(self.init_args + args)

        return f'{self.dataset_cls.__name__}({params})'

    def __dict__(self):
        # 不返回实际的数据集对象，只返回其类型与对应的初始化参数，用于下次初始化。
        return {
            'dataset_cls': self.dataset_cls,
            'init_args': self.init_args,
            'init_kwargs': self.init_kwargs,
        }


def load(task: DataLoadingTask):
    return task.load()


def parallel_loading_datasets(tasks: List[DataLoadingTask], max_workers: int = None):
    if len(tasks) == 0:
        return
    if max_workers is None:
        # 进程数 = min{CPU数, 实际任务数}
        max_workers = min(os.cpu_count(), len(tasks))
    with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        # pool.map(load, tasks)
        for task in tasks:
            print(task)
            pool.submit(load, task)
            # pool.submit(task.dataset_cls, task.init_args, task.init_kwargs)


if __name__ == '__main__':
    batch_size = 8
    time_window = timedelta(hours=2)
    hop = 2
    raw_dir = "F:\Python-projects\DatasetAnalysis\data\dataset"
    tasks = [DataLoadingTask(WeiboTopicDataset, raw_dir=raw_dir, hop=hop, time_window=timedelta(hours=1),
                             min_cascade_length=20)
        , DataLoadingTask(WeiboTopicDataset, raw_dir=raw_dir, hop=hop, time_window=timedelta(hours=2),
                          min_cascade_length=20)
        , DataLoadingTask(WeiboTopicDataset, raw_dir=raw_dir, hop=hop, time_window=timedelta(hours=4),
                          min_cascade_length=20)
        , DataLoadingTask(WeiboTopicDataset, raw_dir=raw_dir, hop=hop, time_window=timedelta(hours=12),
                          min_cascade_length=20)
        , DataLoadingTask(WeiboTopicDataset, raw_dir=raw_dir, hop=hop, time_window=timedelta(hours=24),
                          min_cascade_length=20)
             ]
    parallel_loading_datasets(tasks)
    # tasks[1].load()
