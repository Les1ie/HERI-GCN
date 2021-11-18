import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn import HeteroGraphConv
from dgl.utils import expand_as_pair
from torchnlp.nn.attention import Attention



class HomoAttentionConv(nn.Module):
    """
    对异质图中，每一类关系所带代表的子图进行通用的 Attention GCN 处理。
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True, ):
        super(HomoAttentionConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._bias = bias
        self.attention_mlp = nn.Linear(in_feats, 2 * in_feats, bias=False)
        self.agg_mlp = nn.Linear(in_feats, in_feats, bias=False)
        self.agg_neis = nn.Sequential(
            nn.Linear(2 * in_feats, in_feats, bias=False),
            nn.ReLU()
        )
        self.attention = Attention(2 * in_feats, attention_type='dot')
        self.node_readout = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.edge_readout = nn.Linear(3 * in_feats, out_feats, bias=bias)
        self._edge_features = None

    @property
    def edge_features(self):
        return self._edge_features

    def _reduce(self, nodes):
        """
        将 `[n,d_n]` 形状的节点特征进行广播后，与 `[n,N,d]` 形状边特征连接，用于 Attention 计算。
        其中 `N` 为当前节点的度数，`d` 分别表示点和边特征的维度，二者相等。
        :param nodes: 按照度数分批的节点集合
        :return: ndata['agg'] field with shape [n, 2*d]
        """
        n = len(nodes)
        N = nodes.mailbox['feats'].shape[1]
        node_feats = nodes.data['feats']  # shape: [n, d]

        # concatenates with edge features
        node_feats_expended = torch.unsqueeze(node_feats, 1)  # shape: [n, 1, d]
        node_feats_expended = node_feats_expended.repeat(1, N, 1)  # shape: [n, N, d]
        H = torch.cat([node_feats_expended, nodes.mailbox['feats']], 2)  # shape: [n, N, 2*d]

        # transform feature size to 2*d
        node_feats = self.attention_mlp(node_feats)  # shape: [n, 2*d]
        node_feats = torch.unsqueeze(node_feats, 1)  # shape: [n, 1, 2*d]

        outputs, weights = self._attention(H, node_feats)
        outputs = self.agg_neis(outputs)
        return {'agg': outputs}

    def _attention(self, H, node_feats):
        # attention
        outputs, weights = self.attention(node_feats, H)
        outputs = torch.squeeze(outputs, 1)
        weights = torch.squeeze(weights, 1)
        return outputs, weights

    def _agg_neighbors(self, nodes):
        ners = nodes.data['agg']
        feats = nodes.data['feats']

        feats = self.agg_mlp(feats)
        out = torch.cat([feats, ners], 1)
        return {'combined_feats': out}

    def _e_cat_u_v(self, edges):
        h_u = edges.src['feats']
        h_v = edges.dst['feats']
        agg = torch.cat([edges.data['feats'], h_u, h_v], 1)
        return {'agg': agg}

    def forward(self, g: dgl.DGLGraph, node_feats):
        # edge_feats = g.edata['feats']
        with g.local_scope():
            srctypes = g.srctypes[0]
            dsttypes = g.dsttypes[0]
            src_feats, dst_feats = node_feats   # adapted for dgl 0.6.0
            g.nodes[srctypes].data['feats'] = src_feats
            g.nodes[dsttypes].data['feats'] = dst_feats
            # hg.update_all(fn.copy_e('feats', 'feats'), self._reduce)
            g.send_and_recv(g.edges()
                            , fn.copy_e('feats', 'feats')
                            , self._reduce
                            , self._agg_neighbors
                            )
            g.apply_edges(self._e_cat_u_v)
            node_feats = self.node_readout(g.dstdata['combined_feats'])
            edge_feats = self.edge_readout(g.edata['agg'])
            self._edge_features = edge_feats
            return node_feats


class HeteroConv(nn.Module):

    def __init__(self, in_feats, out_feats, rel_names):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.rel_names = rel_names
        self.conv = HeteroGraphConv({rel: HomoAttentionConv(in_feats, out_feats,
                                                            # activation=relu,
                                                            # weight=True,
                                                            # bias=True
                                                            )
                                     for rel in rel_names}
                                    , aggregate='sum'
                                    )

    def forward(self, g):
        with g.local_scope():
            node_features = g.ndata['feats']
            if len(g.ntypes) == 1:  # parse tensor of single node type to dict
                node_features = {g.ntypes[0]: node_features}
            x = self.conv(g, node_features)
            if len(g.ntypes) == 1:  # parse dict of single node type to tensor
                x = x.popitem()[1]
        return x

    def update_edges(self, g: dgl.DGLHeteroGraph):
        """
        对于每一类关系卷积核的 `HomoConv.edge_features` 属性所保存的对应关系的边特征，将其更新到原图中，用于下一次卷积
        :param g: 用于聚合的图
        :return: 更新了边特征后的图
        """
        for rel_name, conv in self.conv.mods.items():
            edge_features = conv.edge_features
            g.edges[rel_name].data['feats'] = edge_features
        return g



