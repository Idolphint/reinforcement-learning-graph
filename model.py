import torch
import torch.nn as nn


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
        return torch.sigmoid(end)  # end bxkx5x18


class Critic(nn.Module):
    # critic的本质是自动计算的reward，order model Predict if b is a subgraph of a
    # input: 当前state = [b, big_n, big_n] action=[b, big_n, sub_n], [b, sub_n, sub_n]
    # out: [b, 1, 1]

    def __init__(self, args):
        super(Critic, self).__init__()
        self.margin = args.margin
        self.use_intersection = False
        self.fc1 = nn.Linear(64*64, 64)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, emb_as, emb_bs, similar):  # next_state
        # 在这里a是大图，b是小图
        emb_as_T = torch.transpose(emb_as, 1, 0)
        x = torch.matmul(emb_as_T, similar)
        x = torch.matmul(x, emb_bs)
        x = x.view(-1, 64*64)
        x = self.ReLU(self.fc1(x))
        x = self.fc2(x)

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

    def criterion(self, emb_as, emb_bs, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0);
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        a是大图,b是小图
        """
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
            device=emb_as.device), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss