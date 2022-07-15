import os
import pickle
import random

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import scipy.stats as stats
from dataCenter import GraphSet, facebookGraph
from gcn.common import combined_syn
from gcn.common import feature_preprocess
from gcn.common import utils


def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.
    Used as a helper for DiskDataSource.
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="../data/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="../data/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="../data/cox2", name="COX2")
    elif name == "aids":
        dataset = TUDataset(root="../data/AIDS", name="AIDS")
    elif name == "reddit_binary":
        dataset = TUDataset(root="../data/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb_binary":
        dataset = TUDataset(root="../data/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="../data/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="../data/DBLP_v1", name="DBLP_v1")
    elif name == "ppi":
        dataset = PPI(root="../data/PPI")
    elif name == "qm9":
        dataset = QM9(root="../data/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    elif name == "ltt":
        dataset = GraphSet("../data/graphdb250_node300.data").toNXGraphList()
    elif name == "facebook":
        dataset = [facebookGraph(g) for g in ['0', '107', '348', '414',
                                              '686', '698', '1684', '1912',
                                              '3437', '3980']]
    elif name == "amazon":
        dataset = GraphSet("../data/com-amazon.ungraph.txt", init_method="amazon-sub", max_size=3000).toNXGraphList()
    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name: del graph.name
                graph = pyg_utils.to_networkx(graph).to_undirected()
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task


class DataSource:
    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError


class OTFSynDataSource(DataSource):
    """ On-the-fly generated synthetic data for training the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    with a pre-defined generator (see combined_syn.py).

    DeepSNAP transforms are used to generate the positive and negative examples.
    """

    def __init__(self, max_size=599, min_size=5, n_workers=4,
                 max_queue_size=256, node_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

    def gen_data_loaders(self, size, batch_size, train=True,
                         use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            dataset = combined_syn.get_dataset("graph", size // 2,
                                               np.arange(self.min_size + 1, self.max_size + 1))
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset) if \
                use_distributed_sampling else None
            loaders.append(TorchDataLoader(dataset,
                                           collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i == 0 else
                                           batch_size // 2, sampler=sampler, shuffle=False))
        loaders.append([None] * (size // batch_size))
        return loaders

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
                  train):
        # 在这个函数中，batch_target是一个大图，根据这个大图生成子图，作为正例
        #
        def sample_subgraph(graph, offset=0, use_precomp_sizes=False,
                            filter_negs=False, supersample_small_graphs=False, neg_target=None,
                            hard_neg_idxs=None):
            if neg_target is not None: graph_idx = graph.G.graph["idx"]
            use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"]
                            in hard_neg_idxs)
            done = False
            n_tries = 0
            while not done:
                if use_precomp_sizes:
                    size = graph.G.graph["subgraph_size"]
                else:
                    if train and supersample_small_graphs:
                        sizes = np.arange(self.min_size + offset,
                                          len(graph.G) + offset)
                        ps = (sizes - self.min_size + 2) ** (-1.1)
                        ps /= ps.sum()
                        size = stats.rv_discrete(values=(sizes, ps)).rvs()
                    else:
                        d = 1 if train else 0
                        size = random.randint(self.min_size + offset - d,
                                              max(self.min_size+offset+1, len(graph.G) // 2 + offset))
                _, neigh = utils.sample_neigh([graph.G], size)

                if self.node_anchored:
                    anchor = neigh[0]
                    for v in graph.G.nodes:
                        graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                                                            anchor == v else torch.zeros(1))
                        # print(v, graph.G.nodes[v]["node_feature"])

                neigh = graph.G.subgraph(neigh)
                if use_hard_neg and train:
                    neigh = neigh.copy()
                    if random.random() < 1.0 or not self.node_anchored:  # add edges
                        non_edges = list(nx.non_edges(neigh))
                        if len(non_edges) > 0:
                            for u, v in random.sample(non_edges, random.randint(1,
                                                                                min(len(non_edges), 5))):
                                neigh.add_edge(u, v)
                    else:  # perturb anchor标定anchor点为1
                        anchor = random.choice(list(neigh.nodes))
                        for v in neigh.nodes:
                            neigh.nodes[v]["node_feature"] = (torch.ones(1) if
                                                              anchor == v else torch.zeros(1))

                if (filter_negs and train and len(neigh) <= 6 and neg_target is
                        not None):
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        neg_target[graph_idx], neigh)
                    if not matcher.subgraph_is_isomorphic(): done = True
                else:
                    done = True

            return graph, DSGraph(neigh)

        augmenter = feature_preprocess.FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)
        neg_target = batch_neg_target
        # TODO: use hard negs
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
                                          int(len(neg_target.G) * 1 / 2)))

        # hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                     if i not in hard_neg_idxs else g)
             for i, g in enumerate(neg_target.G)])
        # 如果这个图是困难的，那么直接使用neg_target里的子图并随机加边！！，否则随机生成

        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
                                                             hard_neg_idxs=hard_neg_idxs)
        if self.node_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                                                        else torch.zeros(1))
                return g

            neg_target = neg_target.apply_transform(add_anchor)
        pos_target = augmenter.augment(pos_target).to(utils.get_device())
        pos_query = augmenter.augment(pos_query).to(utils.get_device())
        neg_target = augmenter.augment(neg_target).to(utils.get_device())
        neg_query = augmenter.augment(neg_query).to(utils.get_device())
        # print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query


class OTFSynImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly synthetic data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """

    def __init__(self, max_size=59, min_size=5, n_workers=4,
                 max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
                         n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                                                                 or not self.node_anchored else torch.zeros(1))
            return g

        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}".format(str(self.node_anchored),
                                                  self.batch_idx)
        if not train:
            fn = "../"+fn
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        else:
            print("renew data if loaded shouldn't print!!", fn)
            cnt=0
            while True:
                cnt+=1
                graphs_a = graphs_a.apply_transform(add_anchor)
                graphs_b = graphs_b.apply_transform(add_anchor)
                for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                    matcher = nx.algorithms \
                        .isomorphism.GraphMatcher(graph_a, graph_b, node_match=(
                        lambda a, b: (a["node_feature"][0] > 0.5) == (
                                b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                    if matcher.subgraph_is_isomorphic():
                        pos_a.append(graph_a)
                        pos_b.append(graph_b)
                    else:
                        neg_a.append(graph_a)
                        neg_b.append(graph_b)
                if len(pos_a) > 0 or cnt > 10:
                    break
            if not os.path.exists("data/cache") and train:
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)

        # if pos_a:
        pos_a = utils.batch_nx_graphs(pos_a)
        pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b


class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.
    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.
    See the load_dataset function for supported datasets.
    """
    def __init__(self, dataset_name, node_anchored=False, min_size=5,
        max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=1000, min_size=5, seed=None,
        filter_negs=False, sample_method="subgraph-tree"):
        batch_size = a
        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set
        if seed is not None:
            random.seed(seed)
        max_size = min(max_size, max(len(g) for g in graphs) // 2)
        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph, a = utils.sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = utils.sample_neigh([graph], random.randint(min_size,
                    len(graph) - 1))
            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)

        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = utils.sample_neigh(graphs, size)
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic():  # a <= b (b is subgraph of a)
                    continue
            neg_a.append(neigh_a)
            neg_b.append(neigh_b)
        pos_a = utils.batch_nx_graphs(pos_a, anchors=pos_a_anchors if
            self.node_anchored else None)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=pos_b_anchors if
            self.node_anchored else None)
        neg_a = utils.batch_nx_graphs(neg_a, anchors=neg_a_anchors if
            self.node_anchored else None)
        neg_b = utils.batch_nx_graphs(neg_b, anchors=neg_b_anchors if
            self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14})
    data_source = DiskDataSource(dataset_name="aids", node_anchored=True)
    print("load aids!!")
    loaders = data_source.gen_data_loaders(size=100, batch_size=10, train=True)
    print("dataloader finish")
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                                                           batch_neg_target, batch_neg_query, False)
        print("one batch",pos_b)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=None)

    # i = 11
    # neighs = [utils.sample_neigh(train, i) for j in range(10000)]
    # clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
    #     nodes in neighs]
    # path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
    #     for graph, nodes in neighs]
    # #plt.subplot(1, 2, i-9)
    # plt.scatter(clustering, path_length, s=10, label="on_the_fly")
    # plt.legend()
    # plt.savefig("plots/clustering-vs-path-length.png")
