import numpy as np
from dataCenter import *
from map import Map
import json
import os

MIN_DISTANCE_BETWEEN_TWO_POINT = 0.1
max_label = 5


class GNN_env(object):
    def __init__(self, initial_k, mainGraphSet, subGraphSet, gt_json):
        self.__sub = subGraphSet
        self.__origin = mainGraphSet
        self.k = initial_k
        self.result = {}
        self.re_reflect_dict = None
        _, big_node, small_node, small_n, max_l = gt_json.split('_')
        self.big_node = int(big_node[-4:])
        self.small_node = int(small_node[-2:])
        self.small_n = int(small_n[-4:])
        max_label = int(max_l[-7:-5])
        self.gt_json = json.load(open(gt_json, 'r'))

    def add_node2_state(self, offs, offb, subG, mainG, action):
        # state是一个adj+fre的矩阵，将新节点的邻居和fre添加到ori_state里面
        sub_node_list = list(self.result.keys())
        big_node_list = list(self.result.values())
        # sub_node_list.append(action[0])
        # big_node_list.append(action[1])

        subNeighbor = self.__sub.nodelListNeighbor(offs, sub_node_list)
        gNeighbor = self.__origin.nodelListNeighbor(offb, big_node_list)
        selected_samll_graph = self.__sub.getSubMap(offs, sub_node_list+subNeighbor)
        selected_big_graph = self.__origin.getSubMap(offb, big_node_list+gNeighbor)

        selected_samll_graph.mask = len(sub_node_list)
        selected_big_graph.mask = len(big_node_list)

        return selected_samll_graph, selected_big_graph

    def reset(self, offset, offJ):
        print("开始reset")
        self.result = {}
        sub_Vset = self.__sub.curVSet(offJ)
        main_Vset = self.__origin.curVSet(offset)
        gt_dict = self.gt_json[self.small_n*offset+offJ]
        gt_dict = {int(a):b for a,b in gt_dict.items()}
        # 选择一个随机的开始点
        small_idx = np.random.randint(0, len(sub_Vset.keys()))
        for i in main_Vset.keys():
            if self.isMeetRules(small_idx, i, offJ, offset, self.result):
                self.result[small_idx] = i
                break

        selected_samll_node = list(self.result.keys())
        selected_big_node = list(self.result.values())
        subNeighbor = self.__sub.nodelListNeighbor(offJ, selected_samll_node)
        gNeighbor = self.__origin.nodelListNeighbor(offset, selected_big_node)

        selected_samll_graph = self.__sub.getSubMap(offJ, selected_samll_node+subNeighbor)
        selected_big_graph = self.__origin.getSubMap(offset, selected_big_node+gNeighbor)

        selected_samll_graph.mask = len(selected_samll_node)
        selected_big_graph.mask = len(selected_big_node)
        return selected_samll_graph, selected_big_graph

    def preSucc(self, vertexNeighbor, map, type):
        # vertexNeighbor and map can be empty
        if not (isinstance(vertexNeighbor, list) and isinstance(map, list)):
            print("Class Vf preSucc() arguments type error! vertexNeighbor and map expected list!")
            exit()
        if not (type == 0 or type == 1):
            print("Class Vf preSucc() arguments value error! type expected 0 or 1!")

        result = []
        # succ
        if type:
            for vertex in vertexNeighbor:
                if vertex not in map:
                    result.append(vertex)
        # pre
        else:
            for vertex in vertexNeighbor:
                if vertex in map:
                    result.append(vertex)
        return result

    # type = 0, __sub; type = 1, __origin
    def edgeLabel(self, offset, index1, index2, type):

        if type:
            ESet = self.__origin.curVESet(offset)
        else:
            ESet = self.__sub.curVESet(offset)

        return ESet[index1][index2]

    def isMeetRules(self, vs, vb, offs, offb, result):

        '''
        #test usage!
        print "-------------------------------------------"
        print "in isMeetRules() vs: %d, vb: %d" %(vs, vb)点的label相同
        print "in isMeetRules() result: ", result
        print "in isMeetRules() subMap: ", subMap 是result中子图的点
        print "in isMeetRules() gMap: ", gMap 是result中大图的点
        print "in isMeetRules() subMNeighbor: ", subMNeighbor
        print "in isMeetRules() gMNeighbor: ", gMNeighbor
        '''
        subMap = list(result.keys())
        gMap = list(result.values())
        subMNeighbor = self.__sub.nodelListNeighbor(offs, subMap)
        gMNeighbor = self.__origin.nodelListNeighbor(offb, gMap)
        # compare label of vs and vb
        subVSet = self.__sub.curVSet(offs)
        gVSet = self.__origin.curVSet(offb)

        if subVSet[vs] != gVSet[vb]:
            # print "vertex label different!"
            return False

        # notice, when result is empty, first pair should be added when their vertexLabels are the same!
        if not result:
            return True

        vsNeighbor = self.__sub.neighbor(offs, vs)
        vbNeighbor = self.__origin.neighbor(offb, vb)

        vsPre = self.preSucc(vsNeighbor, subMap, 0)
        vsSucc = self.preSucc(vsNeighbor, subMap, 1)
        vbPre = self.preSucc(vbNeighbor, gMap, 0)
        vbSucc = self.preSucc(vbNeighbor, gMap, 1)

        '''
        #test usage!
        print "in isMeetRules() vsNeighbor: ", vsNeighbor
        print "in isMeetRules() vbNeighbor: ", vbNeighbor        
        print "in isMeetRules() vsPre: ", vsPre
        print "in isMeetRules() vbPre: ", vbPre
        print "in isMeetRules() vsSucc: ", vsSucc
        print "in isMeetRules() vbSucc: ", vbSucc
        '''

        # 3,4 rule
        if (len(vsPre) > len(vbPre)):  # 子图的邻居数应该小于等于大图的邻居
            # print "len(vsPre) > len(vbPre)!"
            return False

        for pre in vsPre:
            if pre in result and result[pre] not in vbPre:
                # 小图中的前驱节点应该 对应 大图中的前驱结点
                # print("vsPre not in vbPre!")
                return False
            if pre in result and self.edgeLabel(offs, vs, pre, 0) != self.edgeLabel(offb, vb, result[pre], 1):
                # 前驱结点与当前点的边-label应该一样
                # print "eLabel of vs-pre different with eLabel of vb-result[pre]!"
                return False

        '''   
        if(len(vsSucc) > len(vbSucc)):
            #print "len(vsSucc) > len(vbSucc)!"
            return False

        for succ in vsSucc:
            vertex = self.__sub.curVSet(offs)[succ]
            edge = self.edgeLabel(offs, vs, succ, 0)
            if not self.isMatchInvbSucc(offb, vertex, edge, vb, vbSucc):
                #print "not self.isMatchInvbSucc()"
                return False
        '''

        # 5,6 rules
        len1 = len(set(vsNeighbor) & set(subMNeighbor))  # vs的邻居和子图的所有点的邻居的交 ？？？？？
        len2 = len(set(vbNeighbor) & set(gMNeighbor))
        if len1 > len2:  # 也就是vs的邻居数不能大于vb的邻居数？
            # print("5,6 rules mismatch!大图邻居数太少？")
            return False

        # 7 rule
        # 除去result中的邻居、和vs的前驱，子图的数量不能大于大图的数量
        len1 = len(set(self.__sub.curVSet(offs).keys()) - set(subMNeighbor) - set(vsSucc))
        len2 = len(set(self.__origin.curVSet(offb).keys()) - set(gMNeighbor) - set(vbSucc))
        if len1 > len2:
            # print("7 rule mismatch!除去result中的邻居、和vs的前驱，子图的数量不能大于大图的数量")
            return False

        return True

    def vf2Match(self, offset, offJ, select_idx_b, select_idx_s):
        if not isinstance(self.result, dict):
            print("Class Vf Match() arguments type error! result expected dict!")
        curMap = Map(self.result)
        if curMap.isCovered(self.__sub.curVSet(offJ)):  # 似乎不会来到这句
            print("yes!!! match graph!!!")
            print(self.result)
            return 0

        # 1. 是否连接过的不能再被连接
        # 2. 如何保证选择的action是考虑过子图结构的
        # print("大图被选择的点", select_idx)
        # print("主图的表示为: ", node_representation, graph_adj)
        reward = -1

        # for offb in range(len(self.__sub.curVSet(offJ))):
            # 如果小图中的点没有被选择过
        if (select_idx_s not in self.result.keys()) and self.isMeetRules(select_idx_s, select_idx_b, offJ,
                                offset, self.result):
            self.result[select_idx_s] = select_idx_b
            print("大图被选择的点 %d, 小图被选择的点 %d" % (select_idx_b, select_idx_s))
            # print("distance from 原图到 action", distance_action)
            reward = 1
            # break
        self.result[select_idx_s] = select_idx_b

        return reward

    def step(self, offset, offJ, action, subG, mainG):
        # 已经有结果的需要被屏蔽
        # TODO 这是一次处理1 batch的啊？但是根据调用逻辑，会在step出现的都是一次一个

        r1_reward = self.vf2Match(offset, offJ, action[1], action[0])
        subG, mainG = self.add_node2_state(offJ, offset, subG, mainG, action)

        done = 0
        reward = r1_reward
        curMap = Map(self.result)
        if curMap.isCovered(self.__sub.curVSet(offJ)):
            done = 1
        return subG, mainG, reward, done, r1_reward

if __name__ == '__main__':
    a = np.zeros((3,4,5))
    pos = np.array([[0,2], [1,3]])
    print(pos[:,0])
    pos_w = [range(3), pos[:,0], pos[:,1]]
    a[pos_w] = 1
    print(a)
