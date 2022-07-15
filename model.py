import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import gcn.common.models as gcn_model
from  gcn.config import get_default_config
from util import get_order_model, vis_graph_match, nx2batch


model_args = get_default_config()
order_model = get_order_model(model_args)

class NTN(torch.nn.Module):
    def __init__(self, D, k):
        super(NTN, self).__init__()
        self.k = k
        self.D = D
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.w = torch.nn.Parameter(torch.Tensor(self.k, self.D, self.D))
        self.V = torch.nn.Parameter(torch.Tensor(self.k, 2 * self.D))
        self.b = torch.nn.Parameter(torch.Tensor(self.k, 1, 1))

    def init_parameters(self):
        """
        Initializing weights.全是均匀分布？？？？
        """
        torch.nn.init.xavier_uniform_(self.w)
        torch.nn.init.xavier_uniform_(self.V)
        torch.nn.init.xavier_uniform_(self.b)

    def forward(self, batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
        q_size = len(batch_q_em)
        da_size = len(batch_da_em)
        # print("em shape", q_size, da_size, batch_q_em.shape, batch_da_em.shape)
        batch_q_em_adddim = torch.unsqueeze(batch_q_em, -3)  # batch_q_em_adddim bx1x5xc   torch.tensor
        batch_da_em_adddim = torch.unsqueeze(batch_da_em, -3)  # batch_da_em _adddim bx1x18xc   torch.tensor
        T_batch_da_em_adddim = torch.transpose(batch_da_em_adddim, -2, -1)  # T_batch_da_em _adddim bx1xcx18   torch.tensor
        # first part
        first = torch.matmul(batch_q_em_adddim, self.w)  # first bxkx5xc   torch.tensor
        first = torch.matmul(first, T_batch_da_em_adddim)  # first bxkx5x18   torch.tensor
        # print("first", first)
        # first part
        # second part
        ed_batch_q_em = torch.unsqueeze(batch_q_em, -1)  # ed_batch_q_em bx5x1xc   torch.tensor
        ed_batch_q_em = ed_batch_q_em.repeat(1, 1, da_size, 1)  # ed_batch_q_em bx5x18xc   torch.tensor
        ed_batch_q_em = ed_batch_q_em.reshape(-1, q_size * da_size, self.D)  # ed_batch_q_em bx90xc

        ed_batch_da_em = torch.unsqueeze(batch_da_em, -3)  # ed_batch_da_em bx1x18xc   torch.tensor
        ed_batch_da_em = ed_batch_da_em.repeat(1, q_size, 1, 1)  # ed_batch_da_em bx5x18xc   torch.tensor
        ed_batch_da_em = ed_batch_da_em.reshape(-1, q_size * da_size, self.D)  # ed_batch_da_em bx90xc

        mid = torch.cat([ed_batch_q_em, ed_batch_da_em], -1)  # mid bx90x2c
        mid = torch.transpose(mid, -1, -2)  # mid bx2cx90
        mid = torch.matmul(self.V, mid)  # mid bxkx90
        mid = mid.reshape(-1, self.k, q_size, da_size)  # mid bxkx5x18
        # print("in NTN second part:", mid)
        # second part
        end = first + mid
        # + self.b
        # print("NTN before sigmoid", end.shape)
        # print("bias", self.b)
        # end = first + self.b  # 由于mid重复的太多，所以这里不用mid了
        return torch.nn.functional.relu(end)#, dim=-1)  # end bxkxsmallxbig


class Critic(nn.Module):
    # critic的本质是自动计算的reward，order model Predict if b is a subgraph of a
    # input: 当前state = [b, big_n, big_n] action=[b, big_n, sub_n], [b, sub_n, sub_n]
    # out: [b, 1, 1]

    def __init__(self, args):
        super(Critic, self).__init__()
        self.margin = args.margin
        self.use_intersection = False
        self.fc1 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, emb_small, emb_big, similar):  # next_state
        #若similar全为1，那么fc计算之前的矩阵相乘不过是计算大图点i的m维特征之和*小图点j的m维特征值和
        #那自然可以认为毫无意义，还不如大图*小图T再与similar拼接呢
        emb_small_T = torch.transpose(emb_small, 1, 0)
        similar_T = torch.transpose(similar, 1,0)

        x = torch.matmul(emb_big, emb_small_T)
        bn, sn = similar_T.shape
        x = torch.cat([x,similar_T],dim=-1)
        x = x.view(-1, 2)
        x = self.relu(self.fc1(x)).view(bn,-1)
        x = pyg_nn.global_sort_pool(x, torch.zeros([bn]).long().to(x.device), k=5)  # 得到5*sn*2的向量？
        if x.shape[1] < 20:
            pad = torch.zeros([1, 20-x.shape[1]]).float().to(x.device)
            x = torch.cat([x,pad], dim=-1).view(-1)
        else:
            x = x.view(-1)[:20]
        x = self.fc2(x).unsqueeze(0)

        # 在这里a是大图，b是小图
        # raw_pred = torch.max(torch.zeros_like(emb_as,
        #                                       device=emb_as.device), emb_bs - emb_as) ** 2
        # # b是a的子图，则raw_pred趋近0， reward趋近1
        # # wrong = torch.max(torch.zeros_like(emb_as, device=emb_as.device), emb_as - emb_bs) ** 2
        # # TODO 如果允许batch了要改为按batch相加
        # if (len(raw_pred.shape) > 1):  # 如果有batch维
        #     raw_pred = 1 - torch.mean(raw_pred, dim=1, keepdim=True)
        #     # print(raw_pred.shape)
        # else:
        #     print("当没batch维得时候，期望是1*64， 或者是64", raw_pred.shape)
        #     raw_pred = 1 - torch.mean(raw_pred)
        return x

    def predict(self, emb_as, emb_bs):
        # 在这里a是大图，b是小图
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, emb_small, emb_big, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0);
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        e = torch.sum(torch.max(torch.zeros_like(emb_small,
            device=emb_small.device), emb_small - emb_big)**2, dim=1)

        margin = self.margin
        e[labels <= 0] = torch.max(torch.tensor(0.0,
            device=emb_small.device), margin - e)[labels <= 0]

        relation_loss = torch.sum(e)

        return relation_loss

class Actor(nn.Module):
    # input shape: batch, big_n, big_n], [batch, sub_n, sub_n], [batch, big_n, big_n]
    # out shape: [batch, big_n, sub_n]
    def __init__(self, device):
        super(Actor, self).__init__()
        self.device = device
        self.emb_model = order_model.emb_model.to(self.device)
        self.margin = 0.1

    def emb_graph(self, graph):
        x, batch = self.emb_model(nx2batch(graph, self.device),
                                  need_pool=False, return_batch=True)

        return x, batch

    def forward(self, subNMNeighbor, gNMNeighbor,
                return_action=False, use_noise=False, noise_clip=0.001, batchify=False):
        ori_fea_s, batch_s = self.emb_graph(subNMNeighbor)
        ori_fea_b, batch_b = self.emb_graph(gNMNeighbor)
        # for i in subNMNeighbor.mask:
        # print(batch_b.shape, ori_fea_b.shape, batch_s.shape, ori_fea_s.shape)
        # print(batch_s)
        if not batchify:
            subNMNeighbor = [subNMNeighbor]
            gNMNeighbor = [gNMNeighbor]
        res_action_list = []
        pred_list = []
        fea_s_list = []
        fea_b_list = []
        ret_action = []
        for i, (subg, gg) in enumerate(zip(subNMNeighbor, gNMNeighbor)):
            fea_s_o = ori_fea_s[batch_s==i]
            fea_b_o = ori_fea_b[batch_b==i]
            fea_s_o[subg.mask, :] = 1.0
            fea_b_o[gg.mask, :] = -1.0
            q_size, c= fea_s_o.shape
            data_size,_ = fea_b_o.shape
            fea_s = fea_s_o.unsqueeze(1)
            fea_s = fea_s.repeat(1,data_size,1).reshape(q_size*data_size, c)
            fea_b = fea_b_o.unsqueeze(0)
            fea_b = fea_b.repeat(q_size,1,1).reshape(q_size*data_size, c)
            pred = torch.sum(torch.max(torch.zeros_like(fea_s,
                                                 device=fea_s.device), fea_s - fea_b) ** 2, dim=1)

            # print(pred.shape, q_size, data_size, q_size*data_size, pred)
            if use_noise:
                noise = torch.randn_like(pred) * noise_clip
                pred += noise
            pred = 1-pred
            for a in subg.mask:
                for b in gg.mask:
                    pred[a*data_size+b] = -1
            action = torch.argmax(pred)
            action = action.item()
            res_action = [0, 0]
            res_action[0] = list(subg.nodes())[action // data_size]
            res_action[1] = list(gg.nodes())[action % data_size]
            res_action_list.append(res_action)
            pred_list.append(pred.reshape(q_size, data_size))
            fea_s_list.append(fea_s_o)
            fea_b_list.append(fea_b_o)
            ret_action = [action // data_size, action % data_size]
        if not batchify:
            res_action_list = res_action_list[0]
            pred_list = pred_list[0]
            fea_s_list = fea_s_list[0]
            fea_b_list = fea_b_list[0]
        if return_action:
            return res_action_list, pred_list, fea_s_list, fea_b_list, ret_action
        return res_action_list, pred_list, fea_s_list, fea_b_list

    def criterion(self, emb_small, emb_big, labels, action):
        """Loss function for order emb.
        """
        emb_small = emb_small[action[0]]
        emb_big = emb_big[action[1]]
        # print(emb_small, emb_big, emb_small.shape, emb_big.shape)
        e = torch.sum(torch.max(torch.zeros_like(emb_small,
            device=emb_small.device), emb_small - emb_big)**2, dim=-1)
        # print(e)
        margin = self.margin
        if labels < 0:
            e = torch.max(torch.tensor(0.0,
                device=emb_small.device), margin - e)

        relation_loss = torch.sum(e)

        return relation_loss
