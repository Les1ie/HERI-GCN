import dgl
import matplotlib.pyplot as plt
import networkx as nx

class HeteroGraphDrawing:
    def __init__(self, g: dgl.DGLHeteroGraph,
                 base_node_size=15, scale_node_size=10, max_node_size=40,
                 max_plot_bridges=2, max_bridge_length=2,
                 options: dict = {},
                 figure_option: dict = {},
                 ):
        self.g = g
        self.base_node_size = base_node_size
        self.scale_node_size = scale_node_size
        self.max_node_size = max_node_size
        self.max_plot_bridges = max_plot_bridges
        self.max_bridge_length = max_bridge_length
        self.number_of_bridge_paths = 0
        self.node2type = {i: None for i in g.nodes().tolist()}
        self.type2node = {}
        self.follow_connection_style = 'arc3, rad = 0.1'
        self.node_pos = {}
        self.options = {
            'follow': {
                'node_alpha': 0.1,
                'node_color': 'gray',
                'edge_alpha': 0.1,
                'edge_color': 'gray',
                'width': 0.5,
                'arrowsize': 5,
                'connectionstyle': self.follow_connection_style
            },
            'repost': {
                'node_size': self.base_node_size + self.scale_node_size
            },
            'bridge': {
                # override options of follow
                '__super__': 'follow',
                'node_alpha': 0.5,
                'node_color': '#4169E1',
                'edge_alpha': 0.7,
                'width': 1.0,
                'edge_color': '#4169E1',
            },
            'origin': {
                # override options of repost
                '__super__': 'repost',
                'node_color': 'red',
            }
        }
        for k, v in options.items():
            if k in self.options.keys():
                self.options[k].update(v)
        self.figure = plt.figure(**figure_option)

    @property
    def default_option(self):
        option = {
            # edge options
            'edge_color': 'orange',
            'edge_alpha': 1,
            'width': 1,
            'arrowsize': 10,
            'connectionstyle': None,
            # node options
            'node_color': 'orange',
            'node_alpha': 1,
            'node_size': self.base_node_size,
        }
        # for repost nodes and edges
        return option

    def shortest_path(self, ng, source, target):
        p = []
        try:
            p = nx.shortest_path(ng, source=source, target=target)
        except:
            pass
        return p

    def parse_path(self, path: list):
        edges = set()
        l = len(path)
        if l and (l-1) <= self.max_bridge_length:
            for i in range(l - 1):
                edges.add((path[i], path[i + 1]))
            self.number_of_bridge_paths += 1
        return edges

    def extract_bridges(self, ng):
        bridge_edges = set()
        bridge_nodes = set()

        # path to edges
        users = list(self.type2node['repost'].union(self.type2node['origin']))
        tot_repost = len(users)
        for i in range(tot_repost):
            for j in range(i + 1, tot_repost):
                p = self.parse_path(self.shortest_path(ng, users[i], users[j]))
                bridge_edges.update(p)
                if self.number_of_bridge_paths >= self.max_plot_bridges:
                    break
                p = self.parse_path(self.shortest_path(ng, users[j], users[i]))
                bridge_edges.update(p)
                if self.number_of_bridge_paths >= self.max_plot_bridges:
                    break
            if self.number_of_bridge_paths >= self.max_plot_bridges:
                break
        for e in bridge_edges:
            bridge_nodes.add(e[0])
            bridge_nodes.add(e[1])
        bridge_nodes.difference_update(users)
        return bridge_nodes, bridge_edges

    def extract_origins(self, ng):
        origin_nodes = set()
        for k,v in ng.in_degree():
            if v == 0:
                origin_nodes.add(k)
        return origin_nodes

    def get_options(self, etype):
        option = self.default_option
        if '__super__' in self.options[etype]:
            sp = self.options[etype]['__super__']
            option.update(self.options[sp])
        option.update(self.options[etype])
        return option

    @staticmethod
    def node_option(option: dict):
        option_keys = ['node_color', 'node_alpha', 'node_size']
        extract_option = HeteroGraphDrawing.extract_option(option, option_keys)
        return extract_option

    @staticmethod
    def edge_option(option: dict):
        option_keys = ['edge_color', 'edge_alpha', 'width', 'arrowsize', 'connectionstyle']
        return HeteroGraphDrawing.extract_option(option, option_keys)

    @staticmethod
    def extract_option(option, option_keys: dict, replace_keys: dict = {'node_alpha': 'alpha', 'edge_alpha': 'alpha'}):
        opt = {replace_keys.get(k, k): option[k] for k in option_keys}
        return opt

    def draw_etype(self, etype):

        # prepare networkx graph
        ng = nx.DiGraph(dgl.to_networkx(dgl.to_homogeneous(dgl.edge_type_subgraph(self.g, [etype]))))
        ng.remove_edges_from(nx.selfloop_edges(ng))
        ng.remove_nodes_from([n for n, d in nx.degree(ng) if d == 0])

        draw_nodes = set(filter(lambda x: self.node2type[x] is None, ng.nodes()))
        draw_edges = set(ng.edges)
        self.update_pos(ng)

        print(f'number of {etype} nodes: {ng.number_of_nodes()}')
        print(f'number of {etype} edges: {ng.number_of_edges()}')

        if etype == 'repost':
            # origin
            origin_nodes = self.extract_origins(ng)
            origin = 'origin'
            draw_nodes.difference_update(origin_nodes)
            self.type2node[origin] = origin_nodes
            for n in filter(lambda x: self.node2type[x] is None, origin_nodes):
                self.node2type[n] = origin
            print(f'number of origin repost nodes: {len(origin_nodes)}')
            nx.draw_networkx_nodes(ng, nodelist=origin_nodes, pos=self.node_pos, label=origin,
                                   **self.node_option(self.get_options(origin)))

        if etype == 'follow':
            # bridge
            bridge_nodes, bridge_edges = self.extract_bridges(ng)
            bridge = 'bridge'
            self.type2node[bridge] = bridge_nodes
            for n in filter(lambda x: self.node2type[x] is None, bridge_nodes):
                self.node2type[n] = bridge
            print(f'number of bridge follow paths: {self.number_of_bridge_paths}')
            print(f'number of bridge follow edges: {len(bridge_edges)}')
            print(f'number of bridge follow nodes: {len(bridge_nodes)}')
            nx.draw_networkx_nodes(ng, nodelist=bridge_nodes, pos=self.node_pos, label=bridge,
                                   **self.node_option(self.get_options(bridge)))
            nx.draw_networkx_edges(ng, edgelist=bridge_edges, pos=self.node_pos, label=bridge,
                                   **self.edge_option(self.get_options(bridge)))
            draw_edges.difference_update(bridge_edges)
            draw_nodes.difference_update(bridge_nodes)
        self.type2node[etype] = draw_nodes
        edge_options = self.edge_option(self.get_options(etype))
        node_options = self.node_option(self.get_options(etype))
        nx.draw_networkx_nodes(ng, pos=self.node_pos, nodelist=draw_nodes, label=etype, **node_options)
        nx.draw_networkx_edges(ng, pos=self.node_pos, edgelist=draw_edges, label=etype, **edge_options)

    def update_pos(self, ng):
        pos = nx.random_layout(ng)
        pos.update(self.node_pos)
        self.node_pos = pos

    def draw(self, draw_etypes=['repost', 'follow'], show=True):
        for i, e in enumerate(draw_etypes):
            self.draw_etype(e)
        plt.legend()
        if show:
            plt.show()


if __name__ == '__main__':
    g = dgl.load_graphs('temp.graph')
    g = g[0][0]
    HeteroGraphDrawing(g).draw()
