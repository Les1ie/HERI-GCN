from collections import Iterable

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch


def collate(x):
    """
    Collate graphs and labels.
    :param x: list of tuple(graph, label).
    :return: batched graph, tensor of labels.
    """
    graphs, labels = zip(*x)
    labels = torch.tensor(labels, dtype=torch.float)
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels


def extract_graph_structure(g: dgl.DGLHeteroGraph, device='cpu', dtype=torch.int32) -> dict:
    """
    Extract graph structure used to construct new graph.
    :param g: heterogeneous graph.
    :return: edge list of each edge type.
    """
    graph_data = {}
    for rel in g.canonical_etypes:
        # t = g.adj(etype=rel).coalesce().indices().to(device=device, dtype=dtype)
        graph_data[rel] = g.edges(etype=rel)
    return graph_data


def gen_node_feature(g: dgl.DGLHeteroGraph, device='cpu', inplace=False):
    # if isinstance(g, dgl.DGLHeteroGraph):
    nx_g = g.edge_type_subgraph(['repost']).to_networkx()
    coreness = nx.core_number(nx_g)
    pagerank = nx.pagerank(nx_g)
    hits, authority_score = nx.hits(nx_g)
    eigenvector_centrality = nx.eigenvector_centrality(nx_g)
    clustering_coefficie = nx.clustering(nx_g)
    dict_feats = [coreness, pagerank, hits, authority_score, eigenvector_centrality, clustering_coefficie]
    feats = list(zip(*[f.values() for f in dict_feats]))
    feats = [list(i) for i in feats]
    feats = torch.tensor(feats, device=device)
    if inplace:
        g.ndata['user'] = feats
    return feats


def initial_features(graph, add_self_loop=True, in_feats=32, device='cpu', etypes=None, ntypes=None):
    if etypes is None:
        etypes = graph.etypes
    elif isinstance(etypes, str):
        etypes = [etypes]
    elif isinstance(etypes, Iterable):
        etypes = list(etypes)
    else:
        raise TypeError("etypes must be single edge type or iterable edge types.")

    if ntypes is None:
        ntypes = graph.ntypes
    elif isinstance(ntypes, str):
        ntypes = [ntypes]
    elif isinstance(ntypes, Iterable):
        ntypes = list(ntypes)
    else:
        raise TypeError("ntypes must be single node type or iterable node types.")

    if add_self_loop:
        for etype in etypes:
            graph = dgl.add_self_loop(graph, etype)
    for ntype in ntypes:
        n = graph.num_nodes(ntype)
        # if ntype == 'user':
        ones = torch.ones([n, 1], device=device)
        sp = [n, in_feats - 1]
        normal_feats = torch.normal(torch.zeros(sp, device=device), torch.ones(sp, device=device))
        graph.nodes[ntype].data['feats'] = torch.cat([ones, normal_feats], dim=1)
    for etype in etypes:
        m = graph.num_edges(etype)
        ones = torch.ones([m, 1], device=device)
        sp = [m, in_feats - 1]
        normal_feats = torch.normal(torch.zeros(sp, device=device), torch.ones(sp, device=device))
        graph.edges[etype].data['feats'] = torch.cat([ones, normal_feats], dim=1)
    return graph

