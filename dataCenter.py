import numpy as np
import networkx as nx
import random


def init_one_file(inputFile):
    with open(inputFile, "r") as fin:
        lineNum = -1
        curVertexSet = {}
        curEdgeSet = []
        for line in fin:
            lineList = line.strip().split(" ")
            if not lineList:
                print
                "Class GraphSet __init__() line split error!"
                exit()
            # a new graph!
            if lineList[0] == 't':
                # write it to graphSet
                if lineNum > -1:
                    currentGraph = (lineNum, curVertexSet, curEdgeSet)
                    # self.__graphSet.append(currentGraph)
                    # self.__vertexSet.append(curVertexSet)
                    # self.__edgeSet.append(curEdgeSet)
                    # print "Class GraphSet __init__  __graphSet: ", self.__graphSet
                    # print "Class GraphSet __init__  __vertexSet: ", self.__vertexSet
                    # print "Class GraphSet __init__  __edgeSet: ", self.__edgeSet
                lineNum += 1
                curVertexSet = {}
                curEdgeSet = []
            elif lineList[0] == 'v':
                if len(lineList) != 3:
                    print
                    "Class GraphSet __init__() line vertex error!"
                    exit()
                curVertexSet[int(lineList[1])] = int(lineList[2])
            elif lineList[0] == 'e':
                if len(lineList) != 4:
                    print
                    "Class GraphSet __init__() line edge error!"
                    exit()
                curEdgeSet.append((int(lineList[1]), int(lineList[2]), 1))  # int(lineList[3])))

            else:
                # empty line!
                continue


def gen_Gset(inF="./data/com-amazon.ungraph.txt"):
    g = fromEdge2Graph(inF)
    M_G = GraphSet([g], init_method="list")
    subG = getSubGraph(g, max_size=50, n=500)
    train_sub = GraphSet(subG[:400], init_method="list")
    test_sub = GraphSet(subG[400:], init_method="list")
    return M_G, train_sub, test_sub


class GraphSet:
    def __init__(self, inF, init_method="all_in_one_file", max_size=500):
        self.__graphSet = []
        print("warning!!!! 当前数据集不考虑边权")
        if init_method == "all_in_one_file":
            try:
                init_one_file(inF)
            except IOError as e:
                print("Class GraphSet __init__() Cannot open Graph file: ", e)
                exit()
        elif init_method == "list":
            self.__graphSet = inF
        elif init_method == "amazon-sub":
            g = fromEdge2Graph(inF)
            self.__graphSet = getSubGraph(g, max_size, n=1000)
        elif init_method == "amazon":
            g = fromEdge2Graph(inF)
            print("check edge", g.edges)
            self.__graphSet.append(g)
        elif init_method == "facebook":
            self.__graphSet = [facebookGraph(g) for g in ['0', '107', '348', '414',
                                              '686', '698', '1684', '1912',
                                              '3437', '3980']]
        elif init_method == "facebook-sub":
            for idx in ['0', '107', '348', '414',
                        '686', '698', '1684', '1912',
                        '3437', '3980']:
                self.__graphSet.extend(getSubGraph(facebookGraph(idx), n=50))
        else:
            raise NotImplementedError

    def graphSet(self):
        return self.__graphSet

    def curVSet(self, offset):
        if offset >= len(self.__graphSet):
            print
            "Class GraphSet curVSet() offset out of index!"
            exit()
        res = {n:self.__graphSet[offset].nodes[n]["node_feature"] for n in self.__graphSet[offset].nodes}
        return res

    def curESet(self, offset, a, b):
        if offset >= len(self.__graphSet):
            print
            "Class GraphSet curESet() offset out of index!"
            exit()
        return self.__graphSet[offset].edges[a,b]["edge_feature"]

    def curVESet(self, offset):

        if offset >= len(self.__graphSet):
            print
            "Class GraphSet curVESet() offset out of index!"
            exit()
        return self.__graphSet[offset].edges.data('edge_feature', default=1)

    def neighbor(self, offset, vertexIndex):
        return list(self.__graphSet[offset].neighbors(vertexIndex))


    def graphNum(self):
        return len(self.__graphSet)

    def nodelListNeighbor(self, offset, nodel_list):
        # 首先获得对照表
        Vset = set(nodel_list)
        for nodei in nodel_list:
            Vset.update(self.neighbor(offset, nodei))

        return list(Vset-set(nodel_list))

    def getSubMap(self, offset, node_list):

        return self.__graphSet[offset].subgraph(node_list)

    def toNXGraphList(self):
        return self.__graphSet


def facebookGraph(idx):
    edge_file = open("./data/facebook/%s.edges" % idx, 'r')
    feat_file = open("./data/facebook/%s.feat" % idx, 'r')
    node = {}
    edge = []
    for line in feat_file.readlines():
        lineList = line.strip().split(" ")
        node[int(lineList[0])] = [int(lineList[k]) for k in range(1,len(lineList))]
    for line in edge_file.readlines():
        lineList = line.strip().split(" ")
        edge.append([int(lineList[0]), int(lineList[1])])
    graph = nx.Graph()
    node_attr_list = [(n, {"node_feature": node[n]}) for n in node.keys()]
    graph.add_nodes_from(node_attr_list)
    graph.add_edges_from([(a, b, {'edge_feature': 1}) for a, b in edge])
    return graph

def fromEdge2Graph(fin):
    edge_file = open(fin, 'r')
    edge = []
    for line in edge_file.readlines():
        lineList = line.strip().split()
        if lineList[0] == "#":
            continue
        edge.append([int(lineList[0]), int(lineList[1])])
    graph = nx.Graph()
    graph.add_edges_from([(a, b, {'edge_feature': 1}) for a, b in edge])
    node_attr_list = [(n, {"node_feature": 1}) for n in graph.nodes]
    graph.add_nodes_from(node_attr_list)
    return graph

def getSubGraph(graph, max_size=500, n=20):
    c_nodes = random.sample(list(graph.nodes), n)
    sub_g = []
    for cn in c_nodes:
        neigh = [cn]
        frontier = list(set(graph.neighbors(cn)) - set(neigh))
        visited = {cn}
        while len(neigh) < max_size and frontier:
            new_node = random.choice(list(frontier))
            # new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) >= 5:
            sub_g.append(graph.subgraph(neigh))
    return sub_g


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    x = torch.Tensor([1e-4, 2e-3, 3e-4, 1e-4])*1000000
    y1 = F.softmax(x)
    print(y1)