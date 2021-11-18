import dgl
import torch


def sum_cat(g: dgl.DGLHeteroGraph) -> torch.Tensor:
    """
    Summing node features and edge features respectively,
     return concatenated summed nodes feature and summed edge feature.
    :param g: graph with features.
    :return: aggregated feature tensor used to compute output.
    """
    node_feats = [torch.unsqueeze(torch.sum(g.nodes[n].data['feats'], 0), 0) for n in g.ntypes]
    edge_feats = [torch.unsqueeze(torch.sum(g.edges[e].data['feats'], 0), 0) for e in g.etypes]
    node_feats = torch.sum(torch.cat(node_feats, 0), 0)
    edge_feats = torch.sum(torch.cat(edge_feats, 0), 0)
    feats = torch.cat([node_feats, edge_feats], 0)
    return feats


def rel_sum_cat(g: dgl.DGLHeteroGraph):
    """
    Combine node feature and edge feature for each relationship respectively, and concatenate it.
    :param g: graph used to prediction.
    :return: aggregated feature tensor used to compute output.
    """
    feats = []
    for s, e, d in g.canonical_etypes:
        f = sum_cat(g[(s, e, d)])
        feats.append(f)
    feats = torch.cat(feats, 0)
    return feats


def time_hid_sum_cat(g: dgl.DGLHeteroGraph):
    """
    Aggregated feature use time node's `hid` feature, which is computed by rnn.
    Graphs in g should have `time` nodes with `hid` feature.
    :param g: heterogeneous graph that time node has 'hid' feature.
    :return: aggregated feature tensor used to compute output.
    """
    if 'time' not in g.ntypes or 'hid' not in g.node['time'].data.keys():
        raise KeyError(f'Graphs in g should have "time" nodes with "hid" feature.')
