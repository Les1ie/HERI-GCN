import os
from datetime import timedelta
from os import path as osp

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data import Subset
from tqdm import tqdm


class WeiboTopicDataset(dgl.data.DGLDataset):

    def __init__(self, name='topic', time_window: timedelta = timedelta(hours=1), hop: int = 2
                 , min_cascade_length: int = 20
                 , max_workers: int = None
                 , force_rebuild_u2id=True
                 , filter_homo_graph=True
                 , time='relative'
                 , on_multi_files=True
                 , as_label='repost'
                 , sample_rate=1
                 , url=None, raw_dir='data\\weibotopic', save_dir='processed'
                 , hash_key=(), force_reload=False, verbose=False):
        """
        Dataset of Sina Weibo topics.
        :param time_window: obverse time window as model input.
        :param hop: hop of neighbor sample in follow network
        :param min_cascade_length: minimum of cascade length, shorter cascade will be ignore.
        :param max_workers: works of
        :param force_rebuild_u2id:
        :param filter_homo_graph:
        :param time: use relative time.
        :param url: ?
        :param raw_dir:
        :param save_dir:
        :param hash_key:
        :param force_reload:
        :param verbose:
        """
        self.time_window = time_window
        self._name = name
        self.hop = hop
        self.min_cascade_length = min_cascade_length
        self.max_workers = max_workers
        self.filter_homo_graph = filter_homo_graph
        self.sample_rate = sample_rate
        self.as_label = as_label
        self.time = time
        self._follow_graph = None
        self._ud_follow_graph = None
        self._force_rebuild_u2id = force_rebuild_u2id
        self._multi_files = None
        self.graph_list = []
        self.graph_file_list = []
        self.label_list = []  # list of popularity(label)
        self.labels = torch.empty(0)  # tensor of labels
        self.with_feats = False
        self.has_processed = False
        self.with_feature = False
        self.on_multi_files = on_multi_files
        super().__init__(self.name, url, raw_dir, save_dir, hash_key, force_reload, verbose)

    def save(self):
        if not self.on_multi_files:
            # save all graphs and labels to single file at once.
            assert len(self.graph_list) > 0 and len(self.label_list) == len(self.graph_list)
            dgl.data.utils.save_graphs(self.save_path
                                       , self.graph_list
                                       , {'label': self.labels})
        self.save_info()

    def save_info(self):
        dgl.data.utils.save_info(self.info_save_path, {'with_feats': self.with_feats,
                                                       'has_processed': self.has_processed, })

    def load(self):
        if not self.on_multi_files:
            # load all graphs and labels at once from single file.
            self.graph_list, labels = dgl.data.utils.load_graphs(self.save_path)
            self.labels = labels['label']
            if self.labels.dtype == torch.int32:
                self.labels = self.labels.to(dtype=torch.float)
        if osp.exists(self.info_save_path):
            infos = dgl.data.utils.load_info(self.info_save_path)
            if 'with_feats' in infos.keys():
                self.with_feats = infos['with_feats']
            if 'has_processed' in infos.keys():
                self.has_processed = infos['has_processed']

    def process_each(self, func, save=False, *args, **kwargs):
        if self.on_multi_files:
            # file paths
            it_list = [osp.join(self.save_dir, fn) for fn in self.multi_files]
        else:
            it_list = self.graph_list

        for i, it in tqdm(enumerate(it_list), desc='Processing each', total=len(self)):
            try:
                g, l = self[i]
                g = func(g, *args, **kwargs)
                if self.on_multi_files:
                    l = torch.tensor(l, dtype=torch.float)
                    if save:
                        dgl.data.utils.save_graphs(it_list[i], g, {'label': l})
                else:
                    self.graph_list[i] = g
            except KeyError as e:
                if self.verbose:
                    print(f'Some error occurred while processing {it}'
                          f', this item will be skipped, see trace back for detail:')
                    print(e.__traceback__)
                continue
            except Exception as e:
                # reprocess csv to graph
                try:
                    file_name = self.multi_files[i].split("\\")[-1]
                    csv_path = osp.join(self.cascade_dir, f'{file_name[:-4]}.csv')
                    df = pd.read_csv(csv_path)
                    df = self.process_dataframe(df)
                    g, is_homo_graph = self.build_graph(df, topic_id=i)
                    g = func(g, *args, **kwargs)
                except Exception as e:
                    if self.verbose:
                        print(f'Some error occurred while processing {it}'
                              f', this item will be skipped, see trace back for detail:')
                        print(e.__traceback__)
                    continue

        if not self.on_multi_files and save:
            self.save()
        self.has_processed = True
        self.save_info()
        return self

    def multi_save_filepath(self, idx):
        return osp.join(self.save_dir, f'{idx}.bin')

    def has_cache(self):
        if self.on_multi_files:
            files = self.multi_files
            return osp.exists(self.save_dir) and len(files)
        else:
            return osp.exists(self.save_path)

    @property
    def multi_files(self):
        if self._multi_files is None:
            self._multi_files = list(
                map(lambda x: osp.join(self.save_dir, x),
                    sorted(filter(lambda x: x.endswith('bin'), os.listdir(self.save_dir))
                           , key=lambda x: int(x[:-4]))))
        return self._multi_files

    @property
    def filename(self):
        as_lable = 'R' if self.as_label == 'repost' else 'U'
        filename = "{name}-{hop}Hop-{time_window}-{min_cascade_length}-{as_label}".format(
            name=self.name, hop=self.hop, time_window=self.time_window, as_label=as_lable
            , min_cascade_length=self.min_cascade_length,
        )
        if self.sample_rate < 1:
            filename += f'[{self.sample_rate}]'
        filename = filename.replace(':', '.')
        return filename

    @property
    def save_filename(self):
        filename = self.filename + '.bin'
        return filename

    @property
    def info_save_filename(self):
        filename = self.filename + '.info'
        return filename

    @property
    def name(self):
        return self._name

    @property
    def raw_dir(self):
        d = super().raw_dir
        return d

    @property
    def raw_path(self):
        return super().raw_dir

    @property
    def save_dir(self):
        if self.on_multi_files:
            d = osp.join(self.raw_dir, self._save_dir, self.save_filename[:-4])
        else:
            d = osp.join(self.raw_dir, self._save_dir)
        if not osp.exists(d):
            os.mkdir(d)
        return d

    @property
    def save_path(self):
        return osp.join(self.save_dir, self.save_filename)

    @property
    def info_save_path(self):
        return osp.join(osp.join(self.raw_dir, self._save_dir), self.info_save_filename)

    @property
    def global_dir(self):
        d = osp.join(self.raw_dir, 'global')
        if not osp.exists(d):
            os.mkdir(d)
        return d

    @property
    def cascade_dir(self):
        d = osp.join(self.raw_dir, f'{self.name}_cascades')
        if not osp.exists(d):
            os.mkdir(d)
        return d

    @property
    def follow_graph(self) -> nx.DiGraph:
        if self._follow_graph is None:
            edge_list_path = osp.join(self.global_dir, f'{self.name}_relationships.txt')
            self._follow_graph = nx.read_edgelist(path=edge_list_path, delimiter=',',
                                                  create_using=nx.DiGraph)
            # reverse graph
            # 将关注网络反向，u关注了v，说明“影响”将从v传播到u。
            self._follow_graph = nx.reverse(self._follow_graph)
        return self._follow_graph

    @property
    def ud_follow_graph(self) -> nx.Graph:
        if self._ud_follow_graph is None:
            self._ud_follow_graph = self.follow_graph.to_undirected()
        return self._ud_follow_graph

    @property
    def force_rebuild_u2id(self):
        return self._force_rebuild_u2id or self._force_reload or not self.has_cache

    def build_indexes(self, *uid_lists, topic_id):
        """
        Map user id to consecutive integers starting from zero.
        :param uid_lists: a sequence of user ids list.
        :param topic_id: topic id, used to cache.
        :return: user id map.
        """
        cache_dir = osp.join(self.save_dir, 'u2id')
        if not osp.exists(cache_dir):
            os.mkdir(cache_dir)
        cache_path = osp.join(cache_dir, f'{topic_id}.pickle')
        has_cached = osp.exists(cache_path)
        if has_cached and not self.force_rebuild_u2id:
            # load cache
            u2id = dgl.data.utils.load_info(cache_path)
        else:
            uids = set()
            for uid_list in uid_lists:
                uids.update(uid_list)
            u2id = {u: i for i, u in enumerate(uids)}
            # save cache
            dgl.data.utils.save_info(cache_path, u2id)
            # with open(cache_path, "wb+") as pf:
            #     pickle.dump(u2id, pf)
        return u2id

    def neighbor_sample(self, uids, ) -> nx.DiGraph:
        # 缓存k阶邻居矩阵的邻接表，以提高读取速度。
        # BFS sampling
        nbrs = set([str(uid) for uid in uids])  # total users
        nbrs_set = nbrs.copy()  # outer users for BFS iteration
        for l in range(self.hop):
            new_nbrs = set()
            for n in nbrs_set:
                if n in self.follow_graph:
                    for nbr in self.follow_graph[n]:
                        new_nbrs.add(nbr)
            nbrs.update(new_nbrs)
            nbrs_set = new_nbrs
        sub_g = self.follow_graph.subgraph(nbrs)
        return sub_g

    def process_time(self, time_list):
        s = min(time_list)
        if self.time == 'relative':
            time_list = list(map(lambda x: x / (1 + s), time_list))  # 防止出现 除0 的情况
        else:
            time_list = list(map(lambda x: 1 + x - s, time_list))
        return time_list

    def add_node_static_features(self, g):
        pass

    def build_graph(self, data: pd.DataFrame, topic_id):
        """
        Build a heterogeneous graph by pd.DataFrame.
        :param topic_id: id of topic, used to cache.
        :param data: pd.DataFrame with columns ['uid', 'origin_uid', 'created_at'].
        :return: heterogeneous graph contains follow and repost relationships.
        """
        assert set(['uid', 'origin_uid', 'created_at']).issubset(set(data.columns)), \
            'DataFrame should contain columns ["uid", "origin_uid", "created_at"].'
        u = data['origin_uid'].astype('int64').astype('str').tolist()
        v = data['uid'].astype('int64').astype('str').tolist()
        t = self.process_time(data['created_at'].tolist())
        follow_graph = self.neighbor_sample(u + v)
        u2id = self.build_indexes(u, v, follow_graph.nodes, topic_id=topic_id)
        repost_edges = (torch.tensor([u2id[i] for i in u], dtype=torch.int32)
                        , torch.tensor([u2id[i] for i in v], dtype=torch.int32))
        is_homo_graph = follow_graph.number_of_edges() == 0
        if not is_homo_graph:
            edges_tensor = torch.tensor(list(map(lambda x: (u2id[x[0]], u2id[x[1]]), follow_graph.edges)),
                                        dtype=torch.int32).T
            follow_edges = (edges_tensor[0], edges_tensor[1])
        else:
            node_set = list(map(lambda x: u2id[x], set(u + v)))
            follow_edges = (torch.tensor(node_set, dtype=torch.int32),
                            torch.tensor(node_set, dtype=torch.int32),)
        hetero_graph = dgl.heterograph({
            ('user', 'repost', 'user'): repost_edges,
            ('user', 'follow', 'user'): follow_edges
        })
        # add repost time attribute
        # todo: 完成手动设置部分节点特征功能
        hetero_graph.edges['repost'].data['time'] = torch.tensor(t).unsqueeze(-1)
        hetero_graph.edges['repost'].data['raw_time'] = torch.tensor([int(i) for i in data['created_at']]).unsqueeze(-1)
        return hetero_graph, is_homo_graph

    def process(self, files=None):
        print(f'Process dataset {self.filename}')
        if files is None:
            files = list(sorted(filter(lambda x: x.endswith('csv')
                                       , os.listdir(self.cascade_dir))
                                , key=lambda x: int(x[:-4])))
        np.random.shuffle(files)
        file_count = int(len(files) * self.sample_rate)
        saved_file_paths = []
        files = files[:file_count]
        for topic_id, f in tqdm(enumerate(files), total=file_count):
            df = pd.read_csv(osp.join(self.cascade_dir, f))
            users = set(df['uid'].to_numpy().tolist())
            user_cnt = len(users)
            cas_len = len(df)
            df = self.process_dataframe(df)
            if self.as_label == 'user':
                ob_label = len(set(df['uid'].to_numpy().tolist()))  # user obversed
                label = user_cnt
            else:
                ob_label = len(df)  # repost obversed
                label = cas_len
            # filter short cascade
            if ob_label < self.min_cascade_length:
                continue
            g, is_homo_graph = self.build_graph(df, topic_id=topic_id)
            if is_homo_graph and self.filter_homo_graph:
                continue
            if self.on_multi_files:
                fp = self.multi_save_filepath(topic_id)
                dgl.data.utils.save_graphs(fp, g, {'label': torch.tensor(label, dtype=torch.float)})
                saved_file_paths.append(fp)
            else:
                self.graph_list.append(g)
                self.label_list.append(label)
            # break  # for debug
        if not self.on_multi_files:
            self.labels = torch.tensor(self.label_list, dtype=torch.float)
        else:
            self._multi_files = saved_file_paths

    def process_dataframe(self, df):
        df['created_at'] = pd.to_datetime(df['created_at'])
        # add self loop to seed users.
        df['origin_uid'].fillna(df['uid'], inplace=True)
        # select users in obverse time window
        start_time = df.iloc[0]['created_at']
        end_time = start_time + self.time_window
        df = df[df['created_at'] < end_time]
        # parse datetime to float
        df['created_at'] = df['created_at'].map(lambda x: x.timestamp())
        df = df.sort_values('created_at')
        return df

    def train_valid_test(self, train: float = 0.7, valid: float = 0.15, test: float = 0.15):
        idx = list(range(self.__len__()))
        np.random.shuffle(idx)
        train_valid, valid_test = int(self.__len__() * train), int(self.__len__() * (train + valid))
        train = Subset(self, idx[:train_valid])
        valid = Subset(self, idx[train_valid:valid_test])
        test = Subset(self, idx[valid_test:])
        return {
            'train': train,
            'valid': valid,
            'test': test,
        }

    def __getitem__(self, idx):
        if self.on_multi_files:
            g, l = dgl.data.load_graphs(self.multi_files[idx])
            return g[0], l['label'].item()
        else:
            return self.graph_list[idx], self.labels[idx]

    def __len__(self):
        if self.on_multi_files:
            files = self.multi_files
            return len(files)
        else:
            return len(self.graph_list)

    def __repr__(self):
        return f'WeiboTopicDataset(time_window={self.time_window},' \
               f'\n\tHop={self.hop},min_cascade_length={self.min_cascade_length})'


def func(h, name='repost'):
    print(f'\nProcess {name} dataset with time window {timedelta(hours=h)}.')
    return WeiboTopicDataset(name=name, raw_dir="F:\Python-projects\DatasetAnalysis\data\dataset",
                             force_reload=True,
                             time_window=timedelta(hours=h))


hours = [24, 2]

if __name__ == '__main__':
    # Test
    for h in hours:
        for dataname in ['repost', 'twitter', 'topic']:
            ds = WeiboTopicDataset(name=dataname,
                                   raw_dir="F:\Python-projects\DatasetAnalysis\data\dataset",
                                   force_reload=False,
                                   hop=1,
                                   sample_rate=1 if dataname == 'twitter' else 0.05,
                                   min_cascade_length=5 if dataname == 'twitter' else 20,
                                   on_multi_files=False,
                                   as_label='user',
                                   time_window=timedelta(hours=h)
                                   )

    # with futures.ProcessPoolExecutor(max_workers=2) as pool:
    #     l = pool.map(func, hours, ['repost', 'topic'])
    # print(list(l))

    # ds = WeiboTopicDataset(name='twitter',
    #                        raw_dir="F:\Python-projects\DatasetAnalysis\data\dataset",
    #                        sample_rate=1,
    #                        force_reload=True,
    #                        hop=1,
    #                        time_window=timedelta(hours=2)
    #                        )
    # ds2 = WeiboTopicDataset(name='repost',
    #                        raw_dir="F:\Python-projects\DatasetAnalysis\data\dataset",
    #                        sample_rate=0.05,
    #                        force_reload=True,
    #                        hop=1,
    #                        time_window=timedelta(hours=2)
    #                        )
    # ds3 = WeiboTopicDataset(name='twitter',
    #                        raw_dir="F:\Python-projects\DatasetAnalysis\data\dataset",
    #                        sample_rate=1,
    #                        force_reload=True,
    #                        hop=1,
    #                        time_window=timedelta(hours=2)
    #                        )
    # g = ds[556]
    # print(g)

    # print("raw_dir", ds.raw_dir)
    # print("raw_path", ds.raw_path)
    # print("save_dir", ds.save_dir)
    # print("save_path", ds.save_path)
    # print("global_dir", ds.global_dir)
    # print("cascade_dir", ds.cascade_dir)
    # print("force_rebuild_u2id", ds.force_rebuild_u2id)
    # print("has_cache", ds.has_cache())
    # print("number of graph", len(ds))
    # print(ds.train_valid_test())
