import argparse
import time
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

from env import GNN_env
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from dataCenter import GraphSet
from graph import GraphSage
# os.environ['CUDA_VISIBLE_DEVICE'] = '1'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v0")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-5, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000, type=int) # replay buffer size ori=50000
parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.1, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
# parser.add_argument('--exploration_noise', default=0.2, type=float) #ori=0.1
parser.add_argument('--max_episode', default=500, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()


# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
f1="data/graphdb_node1000.data"
f2="data/Q_node20_n1000.data"
script_name = os.path.basename(__file__)
#env = gym.make(args.env_name)  # 这里给出了当前状态，action之间的转变等
origin_graph = GraphSet(f1)
sub_graph = GraphSet(f2)

env = GNN_env(1, origin_graph, sub_graph)
V_num = 1000
max_label = 5
# state_dim = V_num*(V_num+max_label+1)
sub_V_num = 20
# sub_state_dim = sub_V_num*(sub_V_num+max_label+1)
feature_dim = 100
min_Val = torch.tensor(1e-7).float().to(device) # min value
directory = './exp' + script_name + args.env_name +\
            '/'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+'/'
os.makedirs(directory, exist_ok=True)
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    输入是embedding：adj+fre，输出是node_num*out_channel。目的是提取图特征
    自然可以对大图小图分别生成一个embedding
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(0))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        @Params: input=feature = [big_n ,big_n]
        adj = action = big_n, samll_n.猜测是adj的维度反了，adj的维度错了？
        """
        # support = 1,1000,100
        # output =
        support = torch.matmul(input, self.weight)
        # print("support: ", support) # 仔细看还是support的竖列太像了，深究起来还是input的横行长得太像了
        # print(support.shape, adj.shape, input.shape)
        output = torch.matmul(adj, support)
        # print("output: ", output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, z, o, u, r, d = [], [], [], [], [], [], []

        for i in ind:
            X, Y, Z, O, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            z.append(np.array(Z, copy=False))
            o.append(np.array(O, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(z), np.array(o), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def NodeNorm(feature):
    # feature = x, x, node_n, feature_n
    eps = 1e-10
    mean = torch.mean(feature, dim=-2, keepdim=True)
    std = (torch.var(feature, dim=-2, keepdim=True)+eps).sqrt()
    feature = (feature - mean) / std
    return feature

class NTN(torch.nn.Module):
    def __init__(self, q_size, da_size, D, k):
        super(NTN, self).__init__()
        self.k = k
        self.D = D
        self.q_size = q_size
        self.da_size = da_size
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
        batch_q_em_adddim = torch.unsqueeze(batch_q_em, 1)  # batch_q_em_adddim bx1x5xc   torch.tensor
        batch_da_em_adddim = torch.unsqueeze(batch_da_em, 1)  # batch_da_em _adddim bx1x18xc   torch.tensor
        T_batch_da_em_adddim = torch.transpose(batch_da_em_adddim, 2, 3)  # T_batch_da_em _adddim bx1xcx18   torch.tensor
        # first part
        first = torch.matmul(batch_q_em_adddim, self.w)  # first bxkx5xc   torch.tensor
        first = torch.matmul(first, T_batch_da_em_adddim)  # first bxkx5x18   torch.tensor
        #到这里横行几乎一样，可以理解为大图的100个点之间的差别太小了吗？
        # first part
        # second part
        ed_batch_q_em = torch.unsqueeze(batch_q_em, 2)  # ed_batch_q_em bx5x1xc   torch.tensor
        ed_batch_q_em = ed_batch_q_em.repeat(1, 1, self.da_size, 1)  # ed_batch_q_em bx5x18xc   torch.tensor
        ed_batch_q_em = ed_batch_q_em.reshape(-1, self.q_size * self.da_size, self.D)  # ed_batch_q_em bx90xc

        ed_batch_da_em = torch.unsqueeze(batch_da_em, 1)  # ed_batch_da_em bx1x18xc   torch.tensor
        ed_batch_da_em = ed_batch_da_em.repeat(1, self.q_size, 1, 1)  # ed_batch_da_em bx5x18xc   torch.tensor
        ed_batch_da_em = ed_batch_da_em.reshape(-1, self.q_size * self.da_size, self.D)  # ed_batch_da_em bx90xc

        mid = torch.cat([ed_batch_q_em, ed_batch_da_em], 2)  # mid bx90x2c
        mid = torch.transpose(mid, 1, 2)  # mid bx2cx90
        mid = torch.matmul(self.V, mid)  # mid bxkx90
        mid = mid.reshape(-1, self.k, self.q_size, self.da_size)  # mid bxkx5x18
        # print("in NTN second part:", mid)
        # second part
        end = first + mid + self.b
        # print("NTN before sigmoid", end)
        # end = first + self.b  # 由于mid重复的太多，所以这里不用mid了
        return torch.sigmoid(end)  # end bxkx5x18


class Actor(nn.Module):
    # input shape: batch, big_n, big_n], [batch, sub_n, sub_n], [batch, big_n, big_n]
    # out shape: [batch, big_n, sub_n]
    def __init__(self, sub_state_dim, feature_dim, k):
        super(Actor, self).__init__()
        self.k = k
        self.gcn_b1 = GraphConvolution(state_dim, feature_dim)
        self.gcn_b2 = GraphConvolution(feature_dim, feature_dim)
        self.gcn_b3 = GraphConvolution(feature_dim, feature_dim)

        self.gcn_s1 = GraphConvolution(sub_state_dim, feature_dim)
        self.gcn_s2 = GraphConvolution(feature_dim, feature_dim)

        self.NTN = NTN(sub_state_dim, state_dim, feature_dim, k)
        self.Con1 = torch.nn.Conv2d(self.k, 1, (1,1))

    def forward(self, state, sub_adj, main_adj):
        y = self.gcn_b1(state, main_adj)
        norm_y = NodeNorm(y)
        bg_first_layer_em = torch.nn.functional.elu(norm_y) + y
        y = self.gcn_b2(bg_first_layer_em, main_adj)
        norm_y = NodeNorm(y)
        bg_second_layer_em = torch.nn.functional.elu(norm_y) + y
        y = self.gcn_b3(bg_second_layer_em, main_adj)
        norm_y = NodeNorm(y)
        third_layer_bg_em = torch.nn.functional.elu(norm_y) #因为是希望寻找大图中最优的点，所以在1000上做softmax

        x = self.gcn_s1(sub_adj, sub_adj)
        norm_x = NodeNorm(x)
        sm_first_layer = torch.nn.functional.elu(norm_x) + x
        x = self.gcn_s2(sm_first_layer, sub_adj)
        norm_x = NodeNorm(x)
        sm_second_layer = torch.nn.functional.elu(norm_x)
        # 计算完embedding，计算相似度，ps embedding是不是应该在外面计算好直接用呢？
        similar_mat = self.NTN(sm_second_layer, third_layer_bg_em)
        # 存在的问题，similar_mat中的第3维为何都一样？？即1000的维度
        # print("similar before softmax: ", similar_mat)
        conv_simi = self.Con1(similar_mat)
        conv_simi = NodeNorm(conv_simi)
        # print("conv:", conv_simi)
        similar_mat = torch.softmax(conv_simi, dim=3)
        similar_mat = torch.squeeze(similar_mat)
        # print("similar after", similar_mat)

        return similar_mat


class Critic(nn.Module):
    # critic的本质是自动计算的reward，
    # input: 当前state = [b, big_n, big_n] action=[b, big_n, sub_n], [b, sub_n, sub_n]
    # out: [b, 1, 1]

    def __init__(self, state_dim, sub_state_dim):
        super(Critic, self).__init__()

        self.gcn1 = GraphConvolution(state_dim, sub_state_dim)  # b*1*n*n->b*1*sub_n*sub_n
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=3)

    def forward(self, state, action, sub_state):
        att_sub = self.gcn1(state, action) #期望通过action对当前state实施注意力，得到sub_n*sub_n
        att_sub = NodeNorm(att_sub) # n*sub_n*sub_n
        att_sub = torch.nn.functional.relu(att_sub)
        y = self.conv_1(torch.unsqueeze(att_sub - sub_state, 1))
        y = torch.nn.functional.relu(y)

        y = self.conv_2(y)
        q = torch.nn.functional.relu(y)
        q = torch.sum(q)
        return q


class TD3():
    def __init__(self, V_num, sub_V_num, feature_dim, NTN_k):

        self.actor = Actor(V_num, sub_V_num, feature_dim, NTN_k).to(device)
        self.actor_target = Actor(V_num, sub_V_num, feature_dim, NTN_k).to(device)
        self.critic_1 = Critic(V_num, sub_V_num).to(device)
        self.critic_1_target = Critic(V_num, sub_V_num).to(device)
        self.critic_2 = Critic(V_num, sub_V_num).to(device)
        self.critic_2_target = Critic(V_num, sub_V_num).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=args.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=args.learning_rate)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.memory = Replay_buffer(args.capacity)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state, sub_state):
        state = torch.tensor(np.expand_dims(state, 0)).float().to(device)
        sub_state = torch.tensor(np.expand_dims(sub_state, 0)).float().to(device)
        # main_state = torch.tensor(np.expand_dims(main_state, 0)).float().to(device)

        return self.actor(state, sub_state).cpu().data.numpy()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, z, o, u, r, d = self.memory.sample(args.batch_size) #在这里采样，说明这里存储了数据
            state = torch.FloatTensor(x).to(device)
            sub_state = torch.FloatTensor(z).to(device)
            main_state = torch.FloatTensor(o).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            # noise = torch.ones_like(action).data.normal_(0, args.policy_noise).cuda(
            # noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_action = self.actor_target(next_state, sub_state, main_state)  # next action是通过actor类全连接运算计算出的
            # next_action = next_action.clamp(0, self.max_action)  # (-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action, sub_state)
            target_Q2 = self.critic_2_target(next_state, next_action, sub_state)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action, sub_state)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action, sub_state)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state, sub_state, main_state), sub_state).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args.tau) * target_param.data) + args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor-v1.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target-v1.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1-v1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target-v1.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2-v1.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target-v1.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    agent = TD3(V_num, sub_V_num=sub_V_num, feature_dim=feature_dim, NTN_k=3)
    ep_r = 0
    exploration_noise = 0.2

    if args.mode == 'test':
        agent.load()
        for i in range(origin_graph.graphNum()):
            state, sub_state, main_state = env.reset(i,i)
            if state is None:
                print("主图和子图没有匹配上的，应该直接返回")
                continue
            for t in count():
                action = agent.select_action(state, sub_state, main_state)
                # if t<5:
                #     print("test action", action)

                next_state, reward, done, info = env.step(np.float32(action), state, i, j)
                ep_r += reward
                # env.render()
                if done or t == args.max_episode:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        print(origin_graph.graphNum())
        for i in range(origin_graph.graphNum()):
            for j in range(sub_graph.graphNum()):
                # if i>100 and i % 10 == 0:
                #     exploration_noise *= 0.9  # 衰减的noise
                print("================新的图开始了================")
                sub_state, state = env.reset(i,j) # shape of state = n*(n+num_label)这需要确定下来n和num_label，先定n=10, num_label = 5
                if state is None:
                    print("主图和子图没有匹配上的，应该直接返回")
                    continue
                for t in range(args.max_episode):
                    action = agent.select_action(state, sub_state)
                    np.set_printoptions(suppress=True, precision=3)
                    if t%100 == 0:
                        print("this action", action)
                    # action = action.clip(0, max_action)# 设max_edgelabel = max_veterx_label = 20,(env.action_space.low, env.action_space.high)
                    # print("step ", t, "action is ", action)
                    # 上面这句话的意思是把action内的值按照最小为0，最大为xx的方法弄成分段函数
                    next_state, reward, done, info = env.step(action, state, i, j)
                    ep_r += reward
                    # if args.render and i >= args.render_interval : env.render() render就是渲染，用于绘制运动状态，这里可以没有
                    # if reward > 0 or (reward == 0 and np.random.rand()<0.1):  # 增加正样本的比例
                    agent.memory.push((state, next_state, sub_state, main_state, action, reward, np.float16(done)))
                    if j+1 % 10 == 0:
                        print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                    if len(agent.memory.storage) >= args.capacity-1:
                        agent.update(5)

                    state = next_state
                    if done or t == args.max_episode -1:
                        agent.writer.add_scalar('ep_r', ep_r, global_step=i*100+j)
                        if j % args.print_log == 0:
                            print("Ep_i \t{}-{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, j, ep_r, t))
                        if done:
                            print("it perfect done")
                        ep_r = 0
                        break #当空循环了2000次就停下来不再继续填图

                if j % args.log_interval == 0:
                    print("begin save model")
                    agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
