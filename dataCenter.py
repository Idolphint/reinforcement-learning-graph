# -*- coding:utf-8 -*-
# AUTHOR:   yaolili
# FILE:     graph.py
# ROLE:     read graph from inputFile
# CREATED:  2015-11-28 20:55:11
# MODIFIED: 2015-12-04 09:43:50
import numpy as np
class GraphSet:

    def __init__(self, inputFile):
        self.__graphSet = []
        self.__vertexSet = []
        self.__edgeSet = []
        self.__VESet = {}
        self.__adjMatrix = {}
        self.__freEmbedding = {}
        try:
            with open(inputFile, "r") as fin:
                lineNum = -1
                curVertexSet = {}
                curEdgeSet = {}
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
                        curEdgeSet = {}
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
                        edgeKey = str(lineList[1]) + ":" + str(lineList[2])
                        curEdgeSet[edgeKey] = int(lineList[3])
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
        result = [[] for i in range(vertexNum)]

        for key in self.__edgeSet[offset]:
            v1, v2 = key.strip().split(":")
            # print int(v1)
            # print int(v2)
            result[int(v1)].append(key)
            result[int(v2)].append(key)
        self.__VESet[offset] = result
        return result

    def neighbor(self, offset, vertexIndex):
        if offset >= len(self.__vertexSet):
            print
            "Class GraphSet neighbor() offset out of index!"
            exit()

        VESet = self.curVESet(offset)
        aList = VESet[vertexIndex]
        neighborSet = []
        for i in range(len(aList)):
            v1, v2 = aList[i].strip().split(":")
            if int(v1) != vertexIndex:
                neighborSet.append(int(v1))
            elif int(v2) != vertexIndex:
                neighborSet.append(int(v2))
            else:
                exit()
        return neighborSet

    def graphNum(self):
        return len(self.__vertexSet)

    def curAdjMatrix(self, offset):
        if offset in self.__adjMatrix.keys():
            return self.__adjMatrix[offset]
        node_num = len(self.curVSet(offset))
        adj_matrix = np.zeros((node_num, node_num))
        for i in range(node_num):
            neighborSet = self.neighbor(offset, i)
            for item in neighborSet:
                edge = str(i)+":"+str(item)
                label = 0
                if edge in self.curESet((offset)).keys():
                    label = self.curESet(offset)[str(i)+":"+str(item)]
                else:
                    label = self.curESet(offset)[str(item)+":"+str(i)]
                adj_matrix[i, item] = label
        adj_matrix /= 5
        self.__adjMatrix[offset] = adj_matrix

        return adj_matrix

    def kNeighbor(self, offset, nodei, k):
        # 首先获得对照表
        reflect_dict = {0: nodei}
        cnt = 1
        # 获取这个点附近的k阶邻居
        queue = [nodei]
        layer = [0]
        while len(queue) != 0:
            nei_list = self.neighbor(offset, queue[-1])  # 取队头第一个元素
            next_layer = layer[-1] + 1
            if next_layer > k:
                break
            for nei in nei_list:
                if nei not in reflect_dict.keys():
                    reflect_dict[cnt] = nei
                    cnt += 1
                    queue.insert(0, nei)
                    layer.insert(0, next_layer)
                # print("nei of ", queue[0], nei)
            queue.pop()
            layer.pop()

        reverse_reflect_dict = {old_i: new_i for new_i, old_i in reflect_dict.items()}
        main_state = np.zeros((len(reflect_dict), len(reflect_dict)))
        main_state[0, :] = 1
        for new_idx, big_idx in reflect_dict.items():
            neis = self.neighbor(offset, big_idx)
            for nei in neis:
                if nei in reflect_dict.values():
                    nei_i = reverse_reflect_dict[nei]
                    edge = str(big_idx) + ":" + str(nei)
                    if edge in self.curESet(offset).keys():
                        edge_label = self.curESet(offset)[edge]
                    else:
                        edge_label = self.curESet(offset)[str(nei) + ":" + str(big_idx)]
                    main_state[nei_i, new_idx] = main_state[new_idx, nei_i] = edge_label

        return main_state, reflect_dict

    def curFrequencyEmbedding(self, offset):
        if offset in self.__freEmbedding.keys():
            return self.__freEmbedding[offset]
        label_list = np.array(list(self.curVSet(offset).values()))
        max_label = 5
        node_num = len(self.curVSet(offset))
        fre_embedding = np.ones((node_num, max_label+1))
        for i in range(node_num):
            label = label_list[i]
            f_label = len(np.where(label_list == label)[0])
            fre_embedding[i, label] = f_label/node_num
        self.__freEmbedding[offset] = fre_embedding
        return fre_embedding

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    x = torch.Tensor([1e-4, 2e-3, 3e-4, 1e-4])*1000000
    y1 = F.softmax(x)
    print(y1)