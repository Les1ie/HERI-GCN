import dgl
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from torch.nn import Linear, Dropout, ReLU, GRU, LSTM, ModuleDict, LeakyReLU
from torch.optim import Adam
# from pytorch_lightning.metrics import MeanSquaredLogError
from torchmetrics import MeanSquaredLogError, ExplainedVariance, MeanAbsolutePercentageError

from hooks import PopularityPredictorHooks
from nn.conv import HeteroConv
from nn.readout import NodeEdgeSumCatReadout, RelNodeEdgeSumCatReadout
from utils import initial_features, extract_graph_structure
from utils.arg_parse import parse_init_args


class BasePopularityPredictor(
    pl.LightningModule,
    PopularityPredictorHooks,
):
    """最基础的宏观预测模型，统一定义了训练、校验、测试的过程。
    对于不同的数据集、预处理以及参数细节，只需继承该类并重写相关部分即可。
    具体需要订制的部分：
      - 不同的时间节点设置方案。
      - 不同的超参数设置 。
    """

    def __init__(self, in_feats, hid_feats, out_feats, rel_names
                 , learning_rate
                 , dropout_rate
                 , weight_decay
                 , loss=MeanSquaredLogError
                 , gcn_layers=2
                 , readout=NodeEdgeSumCatReadout
                 , activator=LeakyReLU
                 , random_seed=None
                 , require_process=None
                 , *args, **kwargs):
        super().__init__()
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.set_seed(random_seed)

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        if gcn_layers < 2:
            print(f'"gcn_layers" set to 2.')
        self.gcn_layers = max(2, gcn_layers)
        self._require_process = require_process
        if isinstance(activator, type):
            self.activator = activator()
        else:
            self.activator = activator
        # activator_name = self.activator.__class__.__name__
        # if activator_name == 'LeakyReLU':
        #     self._gain = torch.nn.init.calculate_gain('leaky_relu')
        # elif activator_name == 'ReLU':
        #     self._gain = torch.nn.init.calculate_gain('relu')

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        if isinstance(loss, type):
            self.loss = loss()
        else:
            self.loss = loss
        self._rel_names = rel_names
        self._layer_norm = torch.nn.LayerNorm(self.in_feats)
        self.hetero_convs = torch.nn.ModuleList()

        self.hetero_conv1 = HeteroConv(self.in_feats
                                       , self.hid_feats
                                       , self.rel_names)
        self.hetero_conv2 = HeteroConv(self.hid_feats
                                       , self.out_feats
                                       , self.rel_names)
        self.hetero_convs.append(self.hetero_conv1)
        for i in range(self.gcn_layers - 2):
            self.hetero_convs.append(HeteroConv(self.hid_feats, self.hid_feats, self.rel_names))
        self.hetero_convs.append(self.hetero_conv2)
        if isinstance(readout, type):
            self.readout = readout(self.in_feats, self.hid_feats, self.out_feats)
        else:
            self.readout = readout
        self.readout = self.readout.to(device=self.device)

        self._dropout = Dropout(p=self.dropout_rate)
        self._mape = MeanAbsolutePercentageError()
        # init weights
        # self.apply(self.weight_init)
        # 记录字符、数字类型的超参数
        self.save_hyperparameters('in_feats', 'hid_feats', 'gcn_layers', 'out_feats', 'learning_rate', 'dropout_rate',
                                  'weight_decay', 'random_seed')
        # 记录 readout 模块相关的超参数
        # self.save_hyperparameters(self.readout.hyperparameters)
        # todo: 将对象类型的超参数处理为字符（会从checkpoint导致读取模型参数的错误）
        self.save_hyperparameters({
            # 'loss': type(self.loss).__name__,
            # 'activator': type(self.activator).__name__,
            # 'readout': type(self.readout).__name__,
            'loss': loss,
            'activator': activator,
            'readout': readout,
        })

    def set_seed(self, seed=0):
        pl.seed_everything(seed)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = parent_parser.add_argument_group('BasePopularityPredictor')
        parser.add_argument('--gcn_layers', type=int, default=2,
                            help='Number of heterogeneous gcn layer (default %(default)s).')
        parser.add_argument('--in_feats', type=int, default=8,
                            help='Dimension of input features (default %(default)s).')
        parser.add_argument('--hid_feats', type=int, default=16,
                            help='Dimension of hidden features (default %(default)s).')
        parser.add_argument('--out_feats', type=int, default=32,
                            help='Dimension of output features (default %(default)s).')
        parser.add_argument('--weight_decay', type=float, default=5e-3,
                            help='Weight decay of optimizer (default %(default)s).')
        parser.add_argument('--learning_rate', type=float, default=5e-3,
                            help='Learning rate of optimizer (default %(default)s).')
        parser.add_argument('--dropout_rate', type=float, default=0.5,
                            help='Drop rate of features (default %(default)s).')
        parser.add_argument('--random_seed', type=int, default=None,
                            help='Seed of randomness (default %(default)s).')
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return parse_init_args(cls, args, **kwargs)

    @property
    def rel_names(self):
        return self._rel_names

    @property
    def require_process(self):
        if self._require_process is None:
            return True
        return self._require_process

    @require_process.setter
    def require_process(self, require_process):
        self._require_process = require_process

    def forward(self, g):
        # pre-process
        if self.require_process:
            g = self._process_batched_graph(g)
        g = self._on_conv_start(g, self.in_feats)  # hooks
        g = self._conv(g)
        g = self._on_conv_end(g, self.out_feats)  # hooks
        # predict
        popularity = self._predict_batched_graph(g)
        return popularity

    def _conv(self, g):
        for conv in self.hetero_convs:
            in_feats = conv.in_feats
            out_feats = conv.out_feats
            g = self._on_conv_step_start(g, in_feats)  # hooks
            node_feats = conv(g)
            # update features
            g = conv.update_edges(g)
            g.ndata['feats'] = node_feats
            # activate
            g = self.activate_graph_feats(g)
            g = self._on_conv_step_end(g, out_feats)  # hooks

        return g

    def on_train_start(self):
        # self.logger.log_hyperparams(self.hparams, {"MSLE": 0})
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.device != self.device:
            y = y.to(self.device)
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log('train loss', loss.item())
        # self.log('hp_metric', loss.item())
        # self.logger.log_hyperparams(self.hparams, {'MSLE': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.device != self.device:
            y = y.to(self.device)
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log('valid loss', loss.item())
        # self.log('hp_metric', loss.item())
        # self.logger.log_hyperparams(self.hparams, {'MSLE': loss.item()})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if y.device != self.device:
            y = y.to(self.device)
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log('test loss', loss.item())
        self.log('MSLE', loss.item())
        self.logger.log_hyperparams(self.hparams, {'MSLE': loss.item()})
        # self.log('test percentage error', mape.item())
        return loss

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs
        return loss

    def validation_step_end(self, val_step_outputs):
        loss = val_step_outputs
        return loss

    def test_step_end(self, test_step_outputs):
        loss = test_step_outputs
        return loss

    def configure_optimizers(self):
        adam = Adam(self.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        return adam

    def activate_graph_feats(self, g, feats_name='feats'):
        # activate
        if len(g.ntypes) == 1:
            node_feats = self.activator(g.ndata[feats_name])
        else:
            node_feats = {node_type: self.activator(g.nodes[node_type].data[feats_name]) for node_type in g.ntypes}
        edge_feats = {rel_type: self.activator(g.edges[rel_type].data[feats_name]) for rel_type in g.etypes}
        # combine
        g.ndata[feats_name] = node_feats
        g.edata[feats_name] = edge_feats
        return g

    def _process_batched_graph(self, batched_graph):
        """
        :param batched_graph: batched graph.
        :return: batched processed graph.
        """
        graphs = dgl.unbatch(batched_graph)
        graphs = [self.process_graph(g) for g in graphs]
        bg = dgl.batch(graphs)
        return bg

    def process_graph(self, g: dgl.DGLHeteroGraph):
        """
        对输入的仅包含`关注`与`转发`关系的异构图（unbatched）进行预处理。
        可实现的处理：
          1. 特征的初始化
          2. 时间节点的生成
        :param g: 仅包含`关注`与`转发`关系的异构图。
        :return: 处理后的异构图。
        """
        g = initial_features(g, add_self_loop=True, in_feats=self.in_feats, device=self.device
                             , etypes=['repost', 'follow']
                             , ntypes='user')
        return g

    def _predict_batched_graph(self, batched_graph: dgl.DGLHeteroGraph):
        """
        Do prediction for each graph in batched graph.
        :param batched_graph:
        :return: predicted popularity tensor with shape [1,batch_size].
        """
        ubg = dgl.unbatch(batched_graph)
        rst = []
        for g in ubg:
            g = self.on_readout_start(g, self.out_feats)  # hooks
            popularity = self.predict(g)
            popularity = self.on_readout_end(popularity)  # hooks
            rst.append(popularity)
        rst = torch.cat(rst, 0)
        return rst

    def predict(self, g: dgl.DGLHeteroGraph) -> torch.Tensor:
        """
        Aggregate node and edge features to prediction.
        Default using concatenated summing-aggregated node and edge features to prediction by mlp.
        """
        popularity = self.readout(g)
        return popularity

    # def on_conv_step_end(self, g: dgl.DGLHeteroGraph, feats_dim: int) -> dgl.DGLHeteroGraph:
    #     # unbatched drop out
    #     for etype in g.etypes:
    #         efeats = g.edges[etype].data["feats"]
    #         g.edges[etype].data["feats"] = self._dropout(efeats)
    #     for ntype in g.ntypes:
    #         nfeats = g.nodes[ntype].data["feats"]
    #         g.nodes[ntype].data["feats"] = self._dropout(nfeats)
    #     return g

    # def on_conv_start(self, g: dgl.DGLHeteroGraph, feats_dim: int) -> dgl.DGLHeteroGraph:
    #     # unbatched normalize
    #     for etype in g.etypes:
    #         efeats = g.edges[etype].data["feats"]
    #         g.edges[etype].data["feats"] = self._layer_norm(efeats)
    #     for ntype in g.ntypes:
    #         nfeats = g.nodes[ntype].data["feats"]
    #         g.nodes[ntype].data["feats"] = self._layer_norm(nfeats)
    #     return g

    def _on_conv_step_end(self, g, dim):
        # batched dropout
        for etype in g.etypes:
            efeats = g.edges[etype].data["feats"]
            g.edges[etype].data["feats"] = self._dropout(efeats)
        for ntype in g.ntypes:
            nfeats = g.nodes[ntype].data["feats"]
            g.nodes[ntype].data["feats"] = self._dropout(nfeats)

        # unbatch graph.
        ubgs = [self.on_conv_step_end(ubg, dim) for ubg in dgl.unbatch(g)]
        g = dgl.batch(ubgs)
        return g

    def _on_conv_step_start(self, g, dim):
        # unbatch graph
        ubgs = [self.on_conv_step_start(ubg, dim) for ubg in dgl.unbatch(g)]
        g = dgl.batch(ubgs)
        return g

    def _on_conv_start(self, g, dim):
        # batched normalize
        self.norm_feats(g)
        # unbatch graph
        ubgs = [self.on_conv_start(ubg, dim) for ubg in dgl.unbatch(g)]
        g = dgl.batch(ubgs)
        return g

    def norm_feats(self, g):
        for etype in g.etypes:
            efeats = g.edges[etype].data["feats"]
            g.edges[etype].data["feats"] = self._layer_norm(efeats)
        for ntype in g.ntypes:
            nfeats = g.nodes[ntype].data["feats"]
            g.nodes[ntype].data["feats"] = self._layer_norm(nfeats)

    def _on_conv_end(self, g, dim):
        # unbatch graph
        ubgs = [self.on_conv_end(ubg, dim) for ubg in dgl.unbatch(g)]
        g = dgl.batch(ubgs)
        return g


class TimeGNNPopularityPredictor(BasePopularityPredictor):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names
                 , time_nodes, split
                 , readout=RelNodeEdgeSumCatReadout
                 , *args, **kwargs):
        super().__init__(in_feats=in_feats
                         , hid_feats=hid_feats
                         , out_feats=out_feats
                         , rel_names=rel_names
                         , readout=readout
                         , *args, **kwargs)
        self.time_nodes = time_nodes
        self.split = split

        # 记录字符、数字类型的超参数
        self.save_hyperparameters('time_nodes', 'split')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = (super(TimeGNNPopularityPredictor, TimeGNNPopularityPredictor)
                         .add_model_specific_args(parent_parser))
        parser = parent_parser.add_argument_group('TimeGNNPopularityPredictor')
        parser.add_argument('--time_nodes', type=int, default=10, help='Number of time nodes to add in graph.')
        parser.add_argument('--split', type=str, default='time', choices=['time', 'user'],
                            help='The method of time nodes generation (default %(default)s). '
                                 'divide user nodes equally according to number of users in cascade or length '
                                 'of time sequence, and to connect with time nodes.')
        return parent_parser

    @property
    def rel_names(self):
        return self._rel_names + ['repost_at', 'past_to', 'contain']

    def add_time_nodes(self, g: dgl.DGLHeteroGraph):
        if g.device != self.device:
            g = g.to(self.device)
        graph_data = extract_graph_structure(g, self.device)
        # add links between time nodes.
        graph_data[('time', 'past_to', 'time')] = (
            torch.arange(0, self.time_nodes - 1, device=self.device, dtype=torch.int)
            , torch.arange(1, self.time_nodes, device=self.device, dtype=torch.int))

        v, t = torch.unsqueeze(graph_data[('user', 'repost', 'user')][1], 1), g.edges['repost'].data['time']
        # user ids sorted by repost time
        t, indexs = t.sort(dim=0)
        v = v.gather(dim=0, index=indexs)

        # add links between time node and user node.
        if self.split == 'user':
            user_at_time = self.split_by_user(v, t)
        elif self.split == 'time':
            user_at_time = self.split_by_time(v, t)
        else:
            raise AttributeError(f'Attribute "split" must be "user" or "time", could not be "{self.split}".')
        user_at_time = tuple(map(lambda x: x.to(device=self.device), user_at_time))
        graph_data[('user', 'repost_at', 'time')] = user_at_time
        graph_data[('time', 'contain', 'user')] = (user_at_time[1], user_at_time[0])
        time_graph = dgl.heterograph(graph_data, device=g.device)
        time_graph.edges['repost_at'].data['time'] = t
        time_graph.edges['repost'].data['time'] = g.edges['repost'].data['time']
        return time_graph

    def split_by_time(self, users, times):
        """
        Link user node to time node by repost time.
        :param users: user ids.
        :param times: user repost times.
        :return: edges between user nodes and time nodes.
        """
        time_space = torch.linspace(torch.min(times)
                                    , torch.max(times)
                                    , steps=self.time_nodes)
        time_space[-1] = time_space[-1] * 0.9

        edges_list = [], []
        tid = 0
        for i, t in enumerate(times):
            while t > time_space[tid] and tid < self.time_nodes - 1:
                tid += 1
            edges_list[0].append(users[i])
            edges_list[1].append(tid)
        return (torch.tensor(edges_list[0], dtype=torch.int32)
                , torch.tensor(edges_list[1], dtype=torch.int32))

    def split_by_user(self, users, times):
        num_user = len(users)
        v = torch.linspace(0, self.time_nodes - 1, num_user, dtype=torch.int32)
        return users, v

    def _sum_reduce(self, msg='m', output='feats'):
        def f(nodes):
            return {output: torch.sum(nodes.mailbox[msg], dim=1)}

        return f

    def init_time_feats(self, g: dgl.DGLHeteroGraph, device: str = 'cpu'):
        """
        Initialize user-time edge's features and time node's features.
        For user-time edge's features, sampled from normal distribution (T,1) in which T is time stamp.
        For time node's features, summed by it's user-time edges's feature.
        :param g: graph to add time features.
        :param device: features device, 'cuda' or 'cpu'.
        :return: graph with time features.
        """
        repost_etype = ('user', 'repost_at', 'time')
        contain_etype = ('time', 'contain', 'user')
        # int user-time edges feature.
        times = g.edges[repost_etype].data['time']
        edge_feats = [torch.normal(t.item(), 1, [1, self.in_feats]) for t in times]
        edge_feats = torch.cat(edge_feats, 0).to(device=device)
        g.edges[repost_etype].data['feats'] = edge_feats
        g.edges[contain_etype].data['feats'] = edge_feats
        # init time nodes features.
        g.update_all(dgl.function.copy_e('feats', 'time_edge_feats')
                     , self._sum_reduce('time_edge_feats', 'feats')
                     , etype=repost_etype)
        return g

    def process_graph(self, g: dgl.DGLHeteroGraph):
        g = self.add_time_nodes(g)
        g = self.init_time_feats(g, self.device)
        init_etypes = ['past_to']
        init_ntypes = []
        init_etypes.extend(['repost', 'follow'])
        init_ntypes.extend(['user'])
        g = initial_features(g, add_self_loop=True, in_feats=self.in_feats, device=self.device
                             , etypes=init_etypes
                             , ntypes=init_ntypes)
        return g


class TimeRNNPopularityPredictor(TimeGNNPopularityPredictor):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names
                 , readout
                 , heads
                 , rnn, rnn_feats, rnn_layers, rnn_bidirectional
                 , *args, **kwargs):
        super().__init__(in_feats, hid_feats, out_feats, rel_names
                         , readout=readout
                         , *args, **kwargs)
        self.rnn_feats = rnn_feats
        self.rnn_layers = rnn_layers
        self._rnn_layer_normalize = torch.nn.LayerNorm(self.rnn_feats)
        self.rnn_bidirectional = rnn_bidirectional
        # init rnn module
        fs = [self.in_feats, self.hid_feats, self.out_feats]
        if isinstance(rnn, str):
            if rnn not in ['gru', 'lstm']:
                raise ValueError(f'Attribute "rnn" must be "gru" or "lstm", instead of {rnn}, '
                                 f'you can set bidirectional by parameter "rnn_bidirectional".')

            # use ModlueDict to auto synchronise device.
            if rnn == 'gru':
                self.rnn = GRU(self.rnn_feats, self.rnn_feats, self.rnn_layers, batch_first=True
                               , dropout=self.dropout_rate, bidirectional=self.rnn_bidirectional)
            elif rnn == 'lstm':
                self.rnn = LSTM(self.rnn_feats, self.rnn_feats, self.rnn_layers, batch_first=True
                                , dropout=self.dropout_rate, bidirectional=self.rnn_bidirectional)
        else:
            self.rnn = rnn

        # use to transform output hidden state's shape.
        self.rnn_mlp = ModuleDict()
        for dim in fs:
            self.rnn_mlp[f"{dim}in"] = Linear(dim, self.rnn_feats, bias=False)
            self.rnn_mlp[f"{dim}out"] = Linear(self.rnn_num_directions * self.rnn_feats, dim, bias=False)

        # record hyper parameters of string and number
        self.save_hyperparameters('rnn_feats', 'rnn_layers', 'rnn_bidirectional')
        # todo: 将对象类型的超参数处理为字符（会从checkpoint导致读取模型参数的错误）
        # self.save_hyperparameters({"rnn": type(self.rnn).__name__})
        self.save_hyperparameters({"rnn": rnn})

    @property
    def rnn_num_directions(self):
        return 1 + int(self.rnn_bidirectional)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(TimeRNNPopularityPredictor, TimeRNNPopularityPredictor).add_model_specific_args(
            parent_parser)
        parser = parent_parser.add_argument_group('TimeRNNPopularityPredictor')
        parser.add_argument('--rnn', type=str, default='gru', choices=['gru', 'lstm'],
                            help='Type of RNN layer (default %(default)s).')
        parser.add_argument('--rnn_feats', type=int, default=16,
                            help='Shared features dimension for GNN and RNN feature transformation '
                                 '(default %(default)s).')
        parser.add_argument('--rnn_layers', type=int, default=1,
                            help='Recommends 1, '
                                 'feature vectors will be 0 if rnn_layer>1 (default %(default)s).')
        parser.add_argument('--heads', type=int, default=5,
                            help='Number of attention heads (default %(default)s).')
        parser.add_argument('--rnn_bidirectional', type=bool, default=True,
                            choices=[True, False], help='Directional of RNN (default %(default)s).')
        return parent_parser

    def time_rnn(self, g, feats_dim, batched: bool = True, in_norm: bool = True, out_norm: bool = False,
                 hidden_states=None):
        f"""
        :param g: heterogeneous graph with time nodes. 
        :param feats_dim: size of features in {g}.
        :param batched: is {g} a batched graph.
        :param in_norm: normalize time features before input into rnn.
        :param out_norm: normalize time features output from rnn.
        :param hidden_states: hidden states vector, will be used in rnn and will update after rnn, 
                may causing runtime error while backward if buffer is freed.  
        :return: heterogeneous graph after time rnn.
        """
        dim_in = f"{feats_dim}in"
        dim_out = f"{feats_dim}out"
        node_feats = g.nodes['time'].data['feats']
        if batched:
            node_feats = node_feats.reshape(-1, self.time_nodes, feats_dim)
        else:
            node_feats = torch.unsqueeze(node_feats, 0)
        # transform the shape of input features
        node_feats = self.rnn_mlp[dim_in](node_feats)
        if in_norm:
            node_feats = self._rnn_layer_normalize(node_feats)
        if hidden_states is None:
            # use default hidden state to rnn
            f, h = self.rnn(node_feats)
        else:
            f, h = self.rnn(node_feats, hidden_states)
        # transform the shape of features output from rnn
        if hidden_states is not None:
            # update hidden_states parameter.
            hidden_states = h
        if batched:
            f = f.reshape(-1, self.rnn_feats * self.rnn_num_directions)
        else:
            f = torch.squeeze(f, 0)
        if out_norm:
            f = self._rnn_layer_normalize(f)
        f = self.rnn_mlp[dim_out](f)
        g.nodes['time'].data['feats'] = f
        return g

    # unbatched time rnn before convolution start
    # def on_conv_start(self, g: dgl.DGLHeteroGraph, feats_dim: int) -> dgl.DGLHeteroGraph:
    #     g = super(TimeRNNPopularityPredictor, self).on_conv_start(g, feats_dim)
    # g = self.time_rnn(feats_dim, g)
    # return g

    def _on_conv_step_end(self, g, dim):
        # batched rnn to improve effectiveness
        g = self.time_rnn(g, dim, batched=True, in_norm=True)
        g = super(TimeRNNPopularityPredictor, self)._on_conv_step_end(g, dim)
        return g

    # def on_conv_step_end(self, g: dgl.DGLHeteroGraph, feats_dim: int) -> dgl.DGLHeteroGraph:
    #     # unbatched rnn
    #     g = super(TimeRNNPopularityPredictor, self).on_conv_step_end(g, feats_dim)
    #     g = self.time_rnn(feats_dim, g)
    #     return g
