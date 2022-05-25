import torch
from gcn.common import models
from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_gcn_model(ckpt, model):
    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt)
    return model

def get_order_model(args):
    model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.load_state_dict(torch.load("gcn/ckpt/model.pt", map_location='cpu'))
    print("model load from gcn/ckpt/model.pt")
    return model#.cuda()


def nx2batch(graphs, device):
    if not isinstance(graphs, list):
        graphs = [graphs]

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = batch.to(device)

    return batch

def vis_graph_match(subG, mainG, fix):
    # print(subG.mask, subG.nodes)
    subG = subG.subgraph(list(subG.nodes)[:10])
    mainG = mainG.subgraph(list(mainG.nodes)[:10])
    # subG.subgraph(subG.mask), mainG.subgraph(mainG.mask)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)  # 画2行1列个图形的第1个
    ax2 = fig.add_subplot(2, 1, 2)  # 画2行1列个图形的第2个
    edges, weights = zip(*nx.get_edge_attributes(subG, 'edge_feature').items())
    node_color = np.array(list(nx.get_node_attributes(subG, "node_feature").values()))/10.0
    nx.draw(subG, pos=nx.circular_layout(subG), ax=ax1, with_labels=True, edgelist=edges, edge_color=weights, node_color=node_color)

    edges, weights = zip(*nx.get_edge_attributes(mainG, 'edge_feature').items())
    node_color = np.array(list(nx.get_node_attributes(subG, "node_feature").values()))/10.0
    nx.draw(mainG, pos=nx.circular_layout(mainG), ax=ax2, with_labels=True, edgelist=edges, edge_color=weights, node_color=node_color)

    plt.savefig("./visual_RL/%s.jpg"%fix)
    plt.close()
