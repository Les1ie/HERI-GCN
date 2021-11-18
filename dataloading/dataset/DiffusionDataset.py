import os
import os.path as osp
from collections import Iterable
from concurrent import futures
from datetime import timedelta, datetime

import dgl
import networkx as nx
import torch
from dgl.data import DGLDataset
from tqdm import tqdm
from deprecation import deprecated


# @deprecated(details='This dataset is used to read data in "FOREST", no longer maintained.')
class DiffusionDataset(DGLDataset):

    def __init__(self, dataset_name, usage="train", task="macro", time_window: timedelta = timedelta(hours=1)
                 , hop: int = 1
                 , raw_dir='data'
                 , min_cascade_length: int = 0
                 , max_workers: int = None
                 , force_reload=False, verbose=False):

        self._name = dataset_name
        self.task = task
        self.time_window = time_window
        self.usage = usage
        self.hop = hop
        self._verbose = verbose
        self.min_cascade_length = min_cascade_length
        self.max_workers = max_workers

        self.data_dir = osp.join(raw_dir, dataset_name)

        # processed data sava path
        self.processed_dir = osp.join(self.data_dir, "processed")
        format_time_window = str(self.time_window).replace(":", ".")
        self.processed_filename = f"{self.task}-{self.usage}-{self.hop}Hop-{format_time_window}.bin"
        self.processed_file_path = osp.join(self.processed_dir, self.processed_filename)
        # cascadee file path
        self.cascade_train_txt = osp.join(self.data_dir, "cascade.txt")
        self.cascade_test_txt = osp.join(self.data_dir, "cascadetest.txt")
        self.cascade_valid_txt = osp.join(self.data_dir, "cascadevalid.txt")
        self.cascade_txt_map = {'train': self.cascade_train_txt,
                                'valid': self.cascade_valid_txt,
                                'test': self.cascade_test_txt}
        self.cascade_txt = self.cascade_txt_map[self.usage]
        # follow relationship file path
        self.edges_txt = osp.join(self.data_dir, "edges.txt")

        # processed heterogeneous graphs in dataset
        self.datas = []
        self.labels = []

        # not necessary:
        self.cascade_files = [self.cascade_txt, self.cascade_test_txt, self.cascade_valid_txt]
        self.usages = ['test', 'valid', 'train']
        self.tasks = ['micro', 'macro']

        self.ud_global_follow_graph = None
        self.global_follow_graph = None
        super(DiffusionDataset, self).__init__(name=dataset_name,
                                               raw_dir=raw_dir,
                                               force_reload=force_reload,
                                               verbose=verbose)


    def str_to_timestamp(self, s):
        t = int(float(s))
        # parse '1.285452445e+12'
        if 'e' in s:
            t = int(t) // 1000
        return t

    def process_labels(self, cascade: str):
        '''
        Process graph label by cascade.
        For micro prediction problem: label is the next influenced user;
        For macro prediction problem: label is the future popularity (total influenced user numbers).
         In this case, field self.time_window:timedelta is necessary, default is one hour.
         Cascade diffusion whose time span less than self.time_window will be filtered.
        :param cascade: information diffusion cascade, the argument is the line from raw data file,
            formatted as: 'user1,time1 user2,time2, ...'.
        :return:
            cascade: processed cascade, may extract some cascade to compute label.
            label: label for prediction.
        '''
        label = None
        user_time_tuples = [s.split(",") for s in cascade.strip().split(" ")]
        # 原时间格式：'1.285452445e+12'，处理为整数：1285452445000
        user_time_tuples = sorted(map(lambda x: (x[0], self.str_to_timestamp(x[1])), user_time_tuples)
                                  , key=lambda x: x[0])
        start_time = min(user_time_tuples, key=lambda x: x[1])[1]  # 最小的时间戳
        if self.task == 'macro':
            assert isinstance(self.time_window, timedelta), \
                "Field 'time_window' is not instance of datetime.timedelta."
            st = datetime.fromtimestamp(user_time_tuples[0][1])
            # 标签为实际转发量
            label = len(user_time_tuples)
            # 保留观测时间窗口内的级联
            user_time_tuples = filter(lambda x: datetime.fromtimestamp(x[1]) < st + self.time_window
                                      , user_time_tuples)
            # 将时间转换成浮点类型的相对时间：当前时间/初始时间，以避免使用大整数时间戳导致的问题
            user_time_tuples = list(map(lambda x: (x[0], x[1] / start_time), user_time_tuples))
        elif self.task == 'micro':
            next_user, next_time = user_time_tuples[-1]
            # todo: 这里不能直接用原用户id，需要处理。
            # label = next_user
            label = -1
            user_time_tuples = user_time_tuples[:-1]

        # 过滤掉过短的级联
        if len(user_time_tuples) < self.min_cascade_length:
            user_time_tuples = None

        return user_time_tuples, label

    def neighbor_sample(self, id_list: Iterable):
        """
        Sample self.hop hops neighbors from id_list.
        If self.hop is zero, return graph g itself.

        :param g: graph to sample.
        :param id_list: node ids.
        :return: sub-graph from g.
        """
        assert self.hop >= 0, 'Attribute "hop" must be a non-negative integer.'
        if self.hop == 0:
            return self.global_follow_graph
        if self.ud_global_follow_graph is None:
            self.ud_global_follow_graph = self.global_follow_graph.to_undirected()
        nbrs = set(id_list)
        nbrs_set = set()
        for l in range(self.hop):
            for n in nbrs:
                if n in self.ud_global_follow_graph:
                    for nbr in self.ud_global_follow_graph[n]:
                        nbrs_set.add(nbr)
        sub_g = self.global_follow_graph.subgraph(nbrs_set)
        # nbrs = set((nbr for n in nbrs for nbr in ud_g[n]))
        # sub_g = g.subgraph(nbrs)
        return sub_g

    def build_index(self, edges1, edges2):
        idx2u = list(set(edges1[0] + edges1[1] + edges2[0] + edges2[1]))
        u2idx = {u: i for i, u in enumerate(idx2u)}
        edges1_idx = map(lambda x: list(map(lambda y: u2idx[y], x)), edges1)
        edges2_idx = map(lambda x: list(map(lambda y: u2idx[y], x)), edges2)
        return list(edges1_idx), list(edges2_idx)

    def process_follow_edges(self) -> nx.DiGraph:
        follow_graph = nx.DiGraph()
        with open(self.edges_txt, 'r') as f:
            edges = [e.strip().split(",") for e in f.readlines()]
            follow_graph.add_edges_from(edges)

        return follow_graph.reverse(copy=True)

    def process_cascade(self, e: str) -> tuple:
        """
        通过给定的转发级联序列、全局的关注图，构建包含转发、关注关系的异构图以及对应的标签
        :param e: 转发级联序列
        :param global_follow_graph: 全局关注图
        :return: (异构图，标签)
        """

        user_time_tuples, label = self.process_labels(e)
        # if user_time_tuples is None or len(user_time_tuples) < 2:  # 只包含一个用户时，没有意义。
        if user_time_tuples is None:  # 只包含一个用户时，没有意义。
            return None, label
        repost_edges = [[], []]
        time_list = []
        u = None
        for v, t in user_time_tuples:
            if u is None:
                u = v
                # add a self-loop
                repost_edges[0].append(u)
                repost_edges[1].append(u)
            else:
                repost_edges[0].append(u)
                repost_edges[1].append(v)  # repost user
            time_list.append(t)  # repost time

        # Construct heterogeneous graph.
        # Follow relation sub-graph of users in repost cascade.
        sub_follow_graph = self.neighbor_sample(set(repost_edges[0] + repost_edges[1]))
        # tuple list to source node and target node list
        sub_follow_graph_edges = [[], []]
        for i, j in sub_follow_graph.edges:
            sub_follow_graph_edges[0].append(i)
            sub_follow_graph_edges[1].append(j)

        repost_edges, sub_follow_graph_edges = self.build_index(repost_edges, sub_follow_graph_edges)
        sub_follow_graph_edges_tensor = torch.tensor(sub_follow_graph_edges, dtype=torch.int32)  # to tensor
        repost_graph_edges_tensor = torch.tensor(repost_edges, dtype=torch.int32)
        heter_graph = dgl.heterograph(
            {
                ('user', 'repost', 'user'): (repost_graph_edges_tensor[0], repost_graph_edges_tensor[1]),
                ('user', 'follow', 'user'): (sub_follow_graph_edges_tensor[0],
                                             sub_follow_graph_edges_tensor[1])
            }
        )
        # add repost time attribute
        heter_graph.edges['repost'].data['time'] = torch.tensor(time_list).unsqueeze(-1)
        return heter_graph, label

    def process(self):
        if self.verbose:
            print(
                f"Processing {self.name} {self.usage} dataset for {self.task} prediction " +
                f"with hop={self.hop}, time windows={self.time_window}.")
        graphs = []
        self.global_follow_graph = self.process_follow_edges()
        # 该任务属于IO密集型，使用线程池并发编程，提高处理速度
        with open(self.cascade_txt, 'r') as f, futures.ProcessPoolExecutor(self.max_workers) as pool:
            es = f.readlines()
            with tqdm(total=len(es), desc="Processing cascade") as process_bar:
                for e in es:
                    # 处理每一个级联
                    future = pool.submit(self.process_cascade, e)
                    g,l = future.result()
                    if g is not None:
                        graphs.append(g)
                        self.labels.append(l)
        # 读取转发级联，每一行一个转发级联
        # to tensor
        self.labels = torch.tensor(self.labels)
        self.datas = graphs

    def preprocess_all_usage(self):
        for u in self.usages:
            if u != self.usage:
                self.__class__(dataset_name=self._name, usage=u, time_window=self.time_window
                               , force_reload=self._force_reload
                               , verbose=self.verbose, hop=self.hop)

    def has_cache(self):
        return osp.exists(self.processed_dir) and osp.exists(self.processed_file_path)

    def load(self):
        if self.verbose:
            print("Load data from raw files.")
        self.datas, _labels = dgl.load_graphs(self.processed_file_path)
        self.labels = _labels['labels']

    def process_each(self, func=None, *args, **kwargs):
        for i, g in enumerate(self.datas):
            g = func(g, *args, **kwargs)
            self.datas[i] = g
        return self

    def save(self):
        if not osp.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        _labels = {'labels': self.labels}
        dgl.save_graphs(self.processed_file_path, self.datas, _labels)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        return len(self.datas)

    @property
    def etypes(self):
        return [
            ('user', 'repost', 'user'),
            ('user', 'follow', 'user')
        ]


if __name__ == '__main__':
    ds = DiffusionDataset("twitter", verbose=True, force_reload=True,
                          max_workers=1,
                          time_window=timedelta(hours=2))
    ds.preprocess_all_usage()
    print(ds[2][0].edges['repost'].data['time'])
    print(ds[2][1])
    # DiffusionDataset("weibo", verbose=True, force_reload=False).preprocess_all_usage()
    # DiffusionDataset("douban", verbose=True, force_reload=False).preprocess_all_usage()
