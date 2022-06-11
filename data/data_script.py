import numpy as np
import random
import json

def read_data(file_name, max_vertex_num, max_vertex_label):
    graphSet = []
    vertexSet = []
    edgeSet = []
    with open(file_name, 'r') as fin:
        lineNum = -1
        curVertexSet = {}
        curEdgeSet = {}
        for line in fin:
            lineList = line.strip().split(" ")
            if not lineList:
                print("Class GraphSet __init__() line split error!")
                exit()
            # a new graph!
            if lineList[0] == 't':
                # write it to graphSet
                if lineNum > -1:
                    currentGraph = (lineNum, curVertexSet, curEdgeSet)
                    vertex_num = len(curVertexSet)
                    if vertex_num < max_vertex_num and max(curVertexSet) < max_vertex_label:
                        graphSet.append(currentGraph)
                        vertexSet.append(curVertexSet)
                        edgeSet.append(curEdgeSet)
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
                curVertexSet[int(lineList[1])] = lineList[2]
            elif lineList[0] == 'e':
                if len(lineList) != 4:
                    print
                    "Class GraphSet __init__() line edge error!"
                    exit()
                edgeKey = str(lineList[1]) + ":" + str(lineList[2])
                curEdgeSet[edgeKey] = lineList[3]
            else:
                # empty line!
                continue
    return graphSet, vertexSet, edgeSet

def write_data(file_name, graphSet, vertexSet, edgeSet):
    with open(file_name, 'w') as fout:
        out_str = []
        lineNum = 0
        for curGraph, curVertexSet, curEdgeSet in zip(graphSet, vertexSet, edgeSet):
            # print("# t %d\n"%(lineNum))
            out_str.append("# t %d\n" % (lineNum))
            lineNum += 1
            for vertex_idx, vertex_label in curVertexSet.items():
                out_str.append("v %s %s \n" % (vertex_idx, vertex_label))
            for edge, edge_label in curEdgeSet.items():
                v1, v2 = edge.split(":")
                out_str.append("e %s %s %s\n" % (v1, v2, edge_label))
            out_str.append("")
        fout.writelines(out_str)

def gen_data(file_name = "Q", node_num=0, graph_num=1000, edge_pro=0.3):
    max_node_num = 20
    max_node_label = 5
    max_edge_label = 5
    if node_num == 0:
        node_num = np.random.randint(0, max_node_num)
    file_name += "_node%d.data" % (node_num)
    with open(file_name, 'w') as fout:
        out_str = [] #输出的文本
        for i in range(graph_num):
            out_str.append("t # %d\n" % (i)) #第i个图
            for vertex_idx in range(node_num):
                vertex_label = np.random.randint(0, max_node_label)
                out_str.append("v %d %d \n" % (vertex_idx, vertex_label))
            edge_select_prob = np.random.rand() * edge_pro
            for node_i in range(node_num):
                for node_j in range(node_i+1, node_num):
                    if np.random.rand() < edge_select_prob:
                        edge_label = np.random.randint(0, max_edge_label)
                        out_str.append("e %d %d %d\n" % (node_i, node_j, edge_label))
            out_str.append("")
        fout.writelines(out_str)


def sub2Main(sub_file_name):
    out_str = []
    with open(sub_file_name, 'r') as fin:
        for line in fin:
            lineList = line.strip().split(" ")
            if not lineList:
                print("Class GraphSet __init__() line split error!")
                exit()
            out_str.append(line)
            if lineList[0] == 'v' and lineList[1] == '5':
                for i in range(6,20):
                    out_str.append("v %d 0\n" % i)
    with open("main_"+sub_file_name, 'w') as fout:
        fout.writelines(out_str)

def DFS(idx, graph, sub_num):
    node_list = [idx]
    while(len(node_list) < sub_num):
        nei = list(graph[idx].keys())
        if len(nei) == 0:  # 需要回溯
            if len(node_list) <= 1:
                return None
            to_del = node_list[-1]
            # print("当前的node_list", node_list, "要删", to_del, "删之前的nei", graph[node_list[-2]].keys())
            node_list.pop()
            idx = node_list[-1]
            del graph[idx][to_del]
            # print("删去后的邻居", graph[idx].keys())
            continue
        nei_rand = nei[random.randint(0, len(nei)-1)]
        while nei_rand in node_list:
            nei_rand = nei[random.randint(0, len(nei)-1)]
        node_list.append(nei_rand)
        idx = nei_rand
    return node_list

def Main2sub(big_num, sub_num, sub_graph_n, max_label=5, big_edge_pro=0.1):
    # 首先生成子图与大图的对应，并写入gt_file
    if big_edge_pro > 1:  # 如果pro为小数则为概率，否则为度
        big_edge_pro /= big_num

    # 这里给出大图的顶点和label
    out_big_graph_str = []
    big_node_label_dict = {}
    out_big_graph_str.append("t # 0\n")
    for i in range(big_num):
        label = np.random.randint(max_label)
        # label = 0
        big_node_label_dict[i] = label
        out_big_graph_str.append("v %d %d\n"%(i, label))

    #计算大图的边
    edge_big_graph = []
    for i in range(big_num):
        neighbor_4big_node = {}
        for j in range(i):
            if np.random.rand()<big_edge_pro: #假如这是一条边
                label = np.random.randint(max_label)
                neighbor_4big_node[j] = label  #g[i].append(j)
                edge_big_graph[j][i] = label
                out_big_graph_str.append("e %d %d %d\n" % (i, j, label))
        edge_big_graph.append(neighbor_4big_node)
    out_big_graph_str.append("t # 1\n")

    # 生成小图的点
    sub_node_list = []
    for i in range(sub_graph_n):
        graph_idx = random.randint(0, big_num-1)
        node_list = DFS(graph_idx, edge_big_graph.copy(), sub_num)
        if node_list is not None:
            sub_node_list.append(node_list)

    gt = [{s_idx:b_idx for s_idx, b_idx in enumerate(node_list)} for node_list in sub_node_list]
    gt_reverse = [{b_idx:s_idx for s_idx, b_idx in enumerate(node_list)} for node_list in sub_node_list]
    with open("./gt_datanode%d_querynode%d_query%d_maxlabel%d.json"%(big_num, sub_num, len(sub_node_list), max_label), 'w') as fin:
        json.dump(gt, fin, ensure_ascii=True, indent=True)
        print("gt saved!")

    # 给出对应的小图点Label
    sub_node_set = []
    for subgraph_node in gt:
        cur_node_label = {}
        for s_idx, b_idx in subgraph_node.items():
            label = big_node_label_dict[b_idx]
            cur_node_label[s_idx] = label
        sub_node_set.append(cur_node_label)

    # 计算小图的边
    out_samll_graph_str = []
    for i, cur_node_label in enumerate(sub_node_set):
        # 输出小图的点
        out_samll_graph_str.append("t # %d\n" % i)
        for node_idx, label in cur_node_label.items():
            out_samll_graph_str.append("v %d %d\n" % (node_idx, label))

        node_dict_reverse = gt_reverse[i]
        for big_a_idx, sub_a_idx in node_dict_reverse.items():
            a_neighbor_big = edge_big_graph[big_a_idx]
            for big_b, edge_label in a_neighbor_big.items():
                if big_b in node_dict_reverse.keys():
                    small_b = node_dict_reverse[big_b]
                    out_samll_graph_str.append("e %d %d %d\n"%(sub_a_idx, small_b, edge_label))
    out_samll_graph_str.append("t # %d\n" % len(sub_node_set))

    with open("./graphdb_node%d.data"%big_num, 'w') as fb:
        fb.writelines(out_big_graph_str)
    with open("./Q_node%d_n%d.data" % (sub_num, sub_graph_n), 'w') as fs:
        fs.writelines(out_samll_graph_str)

if __name__ == '__main__':
    # gS, vS, eS = read_data("mygraphdb.data", max_vertex_num=100, max_vertex_label=100)
    # write_data("mygraphdb_100_100.data", gS, vS, eS)
    #
    # gS, vS, eS = read_data("Q4.my", max_vertex_num=20, max_vertex_label=20)
    # write_data("Q4_20_20.my", gS, vS, eS)
    # gen_data("Q1000", 6, edge_pro=0.5)
    # sub2Main("Q1000_node6.data")
    Main2sub(747, 29, 500, 10, big_edge_pro=20)

