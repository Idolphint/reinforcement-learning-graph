import numpy as np
from dataCenter import *
from map import Map
import pyhocon
import torch

MIN_DISTANCE_BETWEEN_TWO_POINT = 0.1
max_label = 5


class GNN_env(object):
    def __init__(self, initial_k, mainGraphSet, subGraphSet):
        self.__sub = subGraphSet
        self.__origin = mainGraphSet
        self.k = initial_k
        self.result = {}
        self.re_reflect_dict = None

    def add_node2_state(self, ori_state, idx, adj):
        # state是一个adj+fre的矩阵，将新节点的邻居和fre添加到ori_state里面

        neighbor = np.argwhere(adj[idx] != 0)
        ori_state[idx, idx] = 1.0
        # ori_state[idx] = adj[idx]
        for nei in neighbor:
            # print(nei, ori_state.shape)
            nei = nei[0]
            ori_state[nei][idx] = adj[idx][nei]
            # ori_state[nei] = adj[nei]

        return ori_state

    def reset(self, offset, offJ):
        print("开始reset")
        V_num = len(self.__origin.curVSet(offset))
        # 选择一个随机的开始点
        big_idx = np.random.randint(0, V_num)
        main_state, self.re_reflect_dict = self.__origin.kNeighbor(offset, big_idx, self.k)
        sub_graph_adj = self.__sub.curAdjMatrix(offJ)
        print("reset 结束")
        return sub_graph_adj, main_state

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
        key1 = str(index1) + ":" + str(index2)
        key2 = str(index2) + ":" + str(index1)
        if type:
            ESet = self.__origin.curESet(offset)
        else:
            ESet = self.__sub.curESet(offset)
        if key1 in ESet.keys():
            return ESet[key1]
        else:
            return ESet[key2]

    def isMeetRules(self, v1, v2, i, j, result, subMap, gMap, subMNeighbor, gMNeighbor):

        '''
        #test usage!
        print "-------------------------------------------"
        print "in isMeetRules() v1: %d, v2: %d" %(v1, v2)点的label相同
        print "in isMeetRules() result: ", result
        print "in isMeetRules() subMap: ", subMap
        print "in isMeetRules() gMap: ", gMap
        print "in isMeetRules() subMNeighbor: ", subMNeighbor
        print "in isMeetRules() gMNeighbor: ", gMNeighbor
        '''

        # compare label of v1 and v2
        subVSet = self.__sub.curVSet(i)
        gVSet = self.__origin.curVSet(j)

        if subVSet[v1] != gVSet[v2]:
            # print "vertex label different!"
            return False

        # notice, when result is empty, first pair should be added when their vertexLabels are the same!
        if not result:
            return True

        v1Neighbor = self.__sub.neighbor(i, v1)
        v2Neighbor = self.__origin.neighbor(j, v2)

        v1Pre = self.preSucc(v1Neighbor, subMap, 0)
        v1Succ = self.preSucc(v1Neighbor, subMap, 1)
        v2Pre = self.preSucc(v2Neighbor, gMap, 0)
        v2Succ = self.preSucc(v2Neighbor, gMap, 1)

        '''
        #test usage!
        print "in isMeetRules() v1Neighbor: ", v1Neighbor
        print "in isMeetRules() v2Neighbor: ", v2Neighbor        
        print "in isMeetRules() v1Pre: ", v1Pre
        print "in isMeetRules() v2Pre: ", v2Pre
        print "in isMeetRules() v1Succ: ", v1Succ
        print "in isMeetRules() v2Succ: ", v2Succ
        '''

        # 3,4 rule
        if (len(v1Pre) > len(v2Pre)):  # 子图的邻居数应该小于等于大图的邻居
            # print "len(v1Pre) > len(v2Pre)!"
            return False

        for pre in v1Pre:
            if pre in result and result[pre] not in v2Pre:
                # 小图中的前驱节点应该 对应 大图中的前驱结点
                # print("v1Pre not in v2Pre!")
                return False
            if pre in result and self.edgeLabel(i, v1, pre, 0) != self.edgeLabel(j, v2, result[pre], 1):
                # 前驱结点与当前点的边-label应该一样
                # print "eLabel of v1-pre different with eLabel of v2-result[pre]!"
                return False

        '''   
        if(len(v1Succ) > len(v2Succ)):
            #print "len(v1Succ) > len(v2Succ)!"
            return False

        for succ in v1Succ:
            vertex = self.__sub.curVSet(i)[succ]
            edge = self.edgeLabel(i, v1, succ, 0)
            if not self.isMatchInV2Succ(j, vertex, edge, v2, v2Succ):
                #print "not self.isMatchInV2Succ()"
                return False
        '''

        # 5,6 rules
        len1 = len(set(v1Neighbor) & set(subMNeighbor))  # v1的邻居和子图的所有点的邻居的交 ？？？？？
        len2 = len(set(v2Neighbor) & set(gMNeighbor))
        if len1 > len2:  # 也就是v1的邻居数不能大于v2的邻居数？
            # print("5,6 rules mismatch!大图邻居数太少？")
            return False

        # 7 rule
        # 除去result中的邻居、和v1的前驱，子图的数量不能大于大图的数量
        len1 = len(set(self.__sub.curVSet(i).keys()) - set(subMNeighbor) - set(v1Succ))
        len2 = len(set(self.__origin.curVSet(j).keys()) - set(gMNeighbor) - set(v2Succ))
        if len1 > len2:
            # print("7 rule mismatch!除去result中的邻居、和v1的前驱，子图的数量不能大于大图的数量")
            return False

        return True

    def vf2Match(self, offset, offJ, select_idx_b, select_idx_s, ori_state):
        if not isinstance(self.result, dict):
            print("Class Vf Match() arguments type error! result expected dict!")
        curMap = Map(self.result)
        if curMap.isCovered(self.__sub.curVSet(offJ)):  # 似乎不会来到这句
            print("yes!!! match graph!!!")
            print(self.result)
            return 0, ori_state
        subMNeighbor = curMap.neighbor(offJ, self.__sub, 0, True)  # 根据result选择关于子图的neighbor
        gMNeighbor = curMap.neighbor(offset, self.__origin, 1, True)
        graph_adj = self.__origin.curAdjMatrix(offset)
        V_num = len(graph_adj)
        # 1. 是否连接过的不能再被连接
        # 2. 如何保证选择的action是考虑过子图结构的
        # print("大图被选择的点", select_idx)
        # print("主图的表示为: ", node_representation, graph_adj)
        reward = -1
        state = ori_state

        # for j in range(len(self.__sub.curVSet(offJ))):
            # 如果小图中的点没有被选择过
        if (select_idx_s not in self.result.keys()) and self.isMeetRules(select_idx_s, select_idx_b, offJ,
                                offset, self.result, curMap.subMap(), curMap.gMap(), subMNeighbor, gMNeighbor):
            # 根据实验允许一点对多点会带来不必要的麻烦
            # if j in self.result.keys():#由于多个大图点对应一个小图点是允许的，但是比较复杂故reward=0
            #     print("多个大图点对应一个小图点")
            #     reward = 0
            #     state = self.add_node2_state(ori_state, select_idx, graph_adj)
            #     break
            self.result[select_idx_s] = select_idx_b
            # print("这一步给出的action", action)
            print("大图被选择的点 %d, 小图被选择的点 %d" % (select_idx_b, select_idx_s))
            # print("distance from 原图到 action", distance_action)
            state = self.add_node2_state(ori_state, select_idx_b, graph_adj)
            reward = len(self.result.keys())
            # break
        if reward < 0: #直接选错了应该重新开始
            print("搞错了，重新开始,选的是", select_idx_b, select_idx_s)
            self.result = {}
            state = self.add_node2_state(ori_state, np.random.randint(0, V_num), graph_adj)
            # reward = -1

        return reward, state

    def step(self, action, state, offset, offJ):
        # 已经有结果的需要被屏蔽
        # TODO 这是一次处理1 batch的啊？但是根据调用逻辑，会在step出现的都是一次一个
        # for my_pair in self.used_pair:
        #     if my_pair is None:
        #         break
        #     small_n, big_n = my_pair
        #     action[small_n, big_n] = 0
        for s_node, b_node in self.result.items():
            action[s_node, b_node] = 0
        select_idx = np.unravel_index(action.argmax(), action.shape)
        select_idx_s, select_idx_b = select_idx
        # self.used_pair[self.used_pair_idx] = (select_idx_s, select_idx_b)
        # self.used_pair_idx += 1
        # self.used_pair_idx %= 10
        r1_reward, next_state = self.vf2Match(offset, offJ, select_idx_b, select_idx_s, state)
        # print("check next state", next_state)
        # TODO eva_loss undefined!!
        done = 0
        reward = r1_reward
        curMap = Map(self.result)
        if curMap.isCovered(self.__sub.curVSet(offJ)):
            done = 1
            reward = 10  # 这里完成的reward更高一点
        return next_state, reward, done, r1_reward

if __name__ == '__main__':
    a = np.zeros((3,4,5))
    pos = np.array([[0,2], [1,3]])
    print(pos[:,0])
    pos_w = [range(3), pos[:,0], pos[:,1]]
    a[pos_w] = 1
    print(a)
