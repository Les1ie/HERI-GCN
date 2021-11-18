import torch
from dgl import DGLHeteroGraph
from torch.nn import Sequential, Linear, ReLU, ModuleList, Flatten, LeakyReLU
from torchnlp.nn import Attention

from utils.arg_parse import parse_init_args


class BaseReadout(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return parse_init_args(cls, args, **kwargs)

    @property
    def hyperparameters(self):
        return {}


class NodeEdgeSumCatReadout(BaseReadout):

    def __init__(self, in_feats, hid_feats, out_feats, device='cpu', *args, **kwargs):
        super(NodeEdgeSumCatReadout, self).__init__(in_feats, hid_feats, out_feats, *args, **kwargs)
        self._output = Sequential(Linear(2 * self.out_feats, 1), ReLU())

    def forward(self, g: DGLHeteroGraph):
        node_feats = [torch.unsqueeze(torch.sum(g.nodes[n].data['feats'], 0), 0) for n in g.ntypes]
        edge_feats = [torch.unsqueeze(torch.sum(g.edges[e].data['feats'], 0), 0) for e in g.etypes]
        node_feats = torch.sum(torch.cat(node_feats, 0), 0)
        edge_feats = torch.sum(torch.cat(edge_feats, 0), 0)
        feats = torch.cat([node_feats, edge_feats], 0)
        pupularity = self._output(feats)
        return pupularity


class RelNodeEdgeSumCatReadout(BaseReadout):
    def __init__(self, in_feats, hid_feats, out_feats, *args, **kwargs):
        super(RelNodeEdgeSumCatReadout, self).__init__(in_feats, hid_feats, out_feats, *args, **kwargs)
        self.sum_cat_readout = NodeEdgeSumCatReadout(in_feats, hid_feats, out_feats)
        out_dim = self.out_feats * 2 * len(self.rel_names)
        # self._output = Linear(out_dim, 1)
        self._output = Sequential(Linear(out_dim, 1), ReLU())

    def forward(self, g: DGLHeteroGraph):
        feats = []
        for s, e, d in g.canonical_etypes:
            f = self.sum_cat_readout(g[(s, e, d)])
            feats.append(f)
        feats = torch.cat(feats, 0)
        popularity = self._output(feats)
        return popularity


class TimeMultiAttendReadout(BaseReadout):
    def __init__(self, in_feats, hid_feats, out_feats, time_nodes, heads=3, readout_use='all',
                 save_attention_weights=False, readout_weighted_sum=False,
                 *args, **kwargs):
        super(TimeMultiAttendReadout, self).__init__(in_feats, hid_feats, out_feats, *args, **kwargs)
        self.heads = heads
        self.time_nodes = time_nodes
        self.user_feats = (False if readout_use == 'time' else True)
        self.attend_time_feats = (False if readout_use == 'user' else True)
        self.readout_weighted_sum = readout_weighted_sum
        self.save_weights = save_attention_weights
        self.weights = None
        # inner modules
        self.multi_attention = ModuleList([Attention(self.out_feats)] * self.heads)  # multi head attenion
        self.multi_attention_mlp = Linear(self.heads * self.out_feats, self.out_feats, bias=False)  # heads*dim -> dim
        self._attend_MLP = Sequential(Flatten(), Linear(out_feats * self.time_nodes, 1), LeakyReLU())
        self._output = None
        if self.readout_weighted_sum:
            self._output = Sequential(Linear(2, 2, bias=True), ReLU(), Linear(2, 1, bias=True))
        else:
            self._output = Sequential(Linear(1, 1, bias=True))

    @property
    def hyperparameters(self):
        return {
            'heads': self.heads
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(TimeMultiAttendReadout, TimeMultiAttendReadout).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group('TimeMultiAttendReadout')
        parser.add_argument('--readout_use', type=str, choices=['all', 'user', 'time'],
                            help='the output layer of Popularity Predictor. '
                                 '"user" means only use summed user features to predict; '
                                 '"time" means only use time attention to predict; '
                                 '"all" means use the sum of user features and time attention to predict.')
        parser.add_argument('--save_attention_weights', action='store_true',
                            help='save weights in attention as readout-layer\'s field, '
                                 'use "readout.weights" to get it.')

        parser.add_argument('--readout_weighted_sum', action='store_true',
                            help='Use weighted summation if time factor and user factor as output, '
                                 'the default output is the product of the two factors, '
                                 'it takes effect when "readout_use" is set to "all". ')
        return parent_parser

    def multi_attend(self, query, context):
        weights = []
        multi_attends = []
        for attention_layer in self.multi_attention:
            attention, weight = attention_layer(query, context)
            multi_attends.append(attention)
            weights.append(weight)
        if self.save_weights:
            self.weights = weights
        attend = torch.cat(multi_attends, 2)
        attention = self.multi_attention_mlp(attend)
        return attention

    def forward(self, g: DGLHeteroGraph):
        if self.user_feats:
            # summing aggregated user features
            summed_user_feats = torch.sum(g.nodes['user'].data["feats"])
        if self.attend_time_feats:
            # multi-head attention aggregated time features
            time_feats = torch.unsqueeze(g.nodes['time'].data['feats'], 0)
            user_feats = torch.unsqueeze(g.nodes['user'].data["feats"], 0)
            attention = self.multi_attend(time_feats, user_feats)
            attend_time_feats = self._attend_MLP(attention)
        if self.user_feats and self.attend_time_feats:
            if self.readout_weighted_sum:
                popularity = self._output(torch.cat([summed_user_feats.view([1]), attend_time_feats.view([1])], 0))
            else:
                popularity = summed_user_feats.view([1]) * attend_time_feats.view([1])
        elif self.attend_time_feats:
            popularity = attend_time_feats
        else:
            popularity = summed_user_feats
        popularity = torch.relu(popularity.view([1]))   # ensure the output is not negative
        return popularity
