import numpy as np
import networkx as nx
class GraphSet:

    def __init__(self, inputFile):
        self.__graphSet = []
        self.__vertexSet = []
        self.__edgeSet = []
        self.__VESet = {}
        try:
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
                            self.__graphSet.append(currentGraph)
                            self.__vertexSet.append(curVertexSet)
                            self.__edgeSet.append(curEdgeSet)
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
                        curEdgeSet.append((int(lineList[1]), int(lineList[2]), int(lineList[3])))

                    else:
                        # empty line!
                        continue
        except IOError as e:
            print("Class GraphSet __init__() Cannot open Graph file: ", e)
            exit()

    def graphSet(self):
        return self.__graphSet

    def curVSet(self, offset):
        if offset >= len(self.__vertexSet):
            print
            "Class GraphSet curVSet() offset out of index!"
            exit()

        return self.__vertexSet[offset]

    def curESet(self, offset):
        if offset >= len(self.__edgeSet):
            print
            "Class GraphSet curESet() offset out of index!"
            exit()

        return self.__edgeSet[offset]

    def curVESet(self, offset):

        if offset >= len(self.__vertexSet):
            print
            "Class GraphSet curVESet() offset out of index!"
            exit()
        if offset in self.__VESet.keys():
            return self.__VESet[offset]

        vertexNum = len(self.__vertexSet[offset])
        result = [{} for i in range(vertexNum)]

        for key in self.__edgeSet[offset]:
            v1, v2, e_label = key
            result[v1][v2] = e_label
            result[v2][v1] = e_label
        self.__VESet[offset] = result
        return result

    def neighbor(self, offset, vertexIndex):
        if offset >= len(self.__vertexSet):
            print
            "Class GraphSet neighbor() offset out of index!"
            exit()

        VESet = self.curVESet(offset)
        aList = VESet[vertexIndex]
        neighborSet = list(aList.keys())

        return neighborSet

    def graphNum(self):
        return len(self.__vertexSet)

    def nodelListNeighbor(self, offset, nodel_list):
        # 首先获得对照表
        Vset = set(nodel_list)
        for nodei in nodel_list:
            Vset.update(self.neighbor(offset, nodei))

        return list(Vset-set(nodel_list))

    def getSubMap(self, offset, node_list):
        oriEset = np.array(self.curESet(offset))
        where_pos = np.isin(oriEset[:, 0], node_list) & np.isin(oriEset[:, 1], node_list)
        Eset = oriEset[where_pos]  # n*3

        graph = nx.Graph()
        node_attr_list = [(i, {"node_feature": self.curVSet(offset)[i]})for i in node_list]
        graph.add_nodes_from(node_attr_list)
        graph.add_edges_from([(a,b,{'edge_feature': c}) for a,b,c in Eset.tolist()])

        return graph


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    x = torch.Tensor([1e-4, 2e-3, 3e-4, 1e-4])*1000000
    y1 = F.softmax(x)
    print(y1)