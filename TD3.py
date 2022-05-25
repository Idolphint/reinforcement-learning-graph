import argparse
import time
from collections import OrderedDict
import gcn.common.models as gcn_model
from  gcn.config import get_default_config
from itertools import count
import os
import numpy as np
import json
from env import GNN_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NTN, Critic
from util import nx2batch, get_order_model, vis_graph_match
from config import args
from tensorboardX import SummaryWriter
from dataCenter import GraphSet
import networkx as nx
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print("uding device: ", device)


# Set seeds
# env.seed( args['seed'])
# torch.manual_seed( args['seed'])
# np.random.seed(args['seed'])
# 设置数据集为真实数据集

f1="data/graphdb_node1000.data"
f2="data/Q_node20_n1000.data"
gt_json = "data/gt_datanode1000_querynode20_query1000_maxlabel10.json"
script_name = os.path.basename(__file__)
#env = gym.make(args['env_name'])  # 这里给出了当前状态，action之间的转变等
origin_graph = GraphSet(f1)
sub_graph = GraphSet(f2)

env = GNN_env(1, origin_graph, sub_graph, gt_json)

min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args['env_name'] + \
            '/'+time.strftime('%Y-%m-%d-%H',time.localtime(time.time()))+'/'
os.makedirs(directory, exist_ok=True)
load_dir = './exp' + script_name + args['env_name'] + '/' + args['load'] + '/'
model_args = get_default_config()

order_model = get_order_model(model_args)
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args['capacity']):
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
        x, y, nx, ny, u, r, d = [], [], [], [], [], [], []

        for i in ind:
            X, Y, NX, NY, U, R, D = self.storage[i]
            x.append(X)
            y.append(Y)
            nx.append(NX)
            ny.append(NY)
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return x, y, nx, ny, np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def NodeNorm(feature):
    # feature = x, x, node_n, feature_n
    eps = 1e-10
    mean = torch.mean(feature, dim=-2, keepdim=True)
    std = (torch.var(feature, dim=-2, keepdim=True)+eps).sqrt()
    feature = (feature - mean) / std
    return feature


class Actor(nn.Module):
    # input shape: batch, big_n, big_n], [batch, sub_n, sub_n], [batch, big_n, big_n]
    # out shape: [batch, big_n, sub_n]
    def __init__(self):
        super(Actor, self).__init__()
        self.order_model = order_model

        self.similar = NTN(k=1,D=64)

    def forward(self, subNMNeighbor, gNMNeighbor, ori_fea_s, ori_fea_b):
        # 先选小图点
        sub_mask_n = subNMNeighbor.mask
        big_mask_n = gNMNeighbor.mask

        fea_s = ori_fea_s[sub_mask_n:, :]
        fea_b = ori_fea_b[big_mask_n:, :]
        # print("check order output", fea_b)
        similar = self.similar(fea_s, fea_b)
        n,_,s,b = similar.shape
        # print(similar)
        similar = similar.reshape(s*b)  # 这里认为k=1

        action = torch.argmax(similar)
        action = action.item()
        res_action = [0,0]
        res_action[0] = list(subNMNeighbor.nodes())[sub_mask_n+(action//b)]
        res_action[1] = list(gNMNeighbor.nodes())[big_mask_n+(action%b)]

        return res_action, similar


class TD3():
    def __init__(self):

        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.critic_1 = Critic(model_args).to(device)
        self.critic_1_target = Critic(model_args).to(device)
        self.critic_2 = Critic(model_args).to(device)
        self.critic_2_target = Critic(model_args).to(device)
        self.emb_model = order_model.emb_model

        self.actor_optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.emb_model.parameters()}], lr=args['learning_rate'])
        self.critic_1_optimizer = optim.Adam([
            {'params':self.critic_1.parameters()},
            {'params':self.emb_model.parameters()}
        ], lr=args['learning_rate'])
        self.critic_2_optimizer = optim.Adam([
            {'params':self.critic_2.parameters()},
            {'params':self.emb_model.parameters()}
        ], lr=args['learning_rate'])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.memory = Replay_buffer(args['capacity'])
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, subNMNeighbor, gNMNeighbor):
        sub_emb = self.emb_model(nx2batch(subNMNeighbor, device), need_pool=False)
        main_emb = self.emb_model(nx2batch(gNMNeighbor, device), need_pool=False)
        return self.actor(subNMNeighbor, gNMNeighbor, sub_emb, main_emb)

    def update(self, num_iteration):
        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, nx, ny, u, r, d = self.memory.sample(args['batch_size']) #在这里采样，说明这里存储了数据
            start_time = time.time()
            time_begin = start_time
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)
            # Compute target Q-value:
            emb_ns = self.emb_model(nx2batch(nx, device))
            emb_nb = self.emb_model(nx2batch(ny, device))
            target_Q1 = self.critic_1_target(emb_nb, emb_ns)
            target_Q2 = self.critic_2_target(emb_nb, emb_ns)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + ((1 - done) * args['gamma'] * target_Q).detach()

            #意为，如果猜对了那reward更大，如果猜错了，reward更小
            # 希望target_Q输出1附近的值，不要太大
            time_end = time.time()
            # print("准备数据耗时：", time_end-time_begin)
            time_begin = time_end
            # Optimize Critic 1:
            emb_s = self.emb_model(nx2batch(x, device))
            emb_b = self.emb_model(nx2batch(y, device))
            current_Q1 = self.critic_1(emb_b, emb_s)
            c1_loss = self.critic_1.criterion(emb_b, emb_s, reward)

            loss_Q1 = torch.mean(F.mse_loss(current_Q1, target_Q)+c1_loss)
            # print("loss Q1", loss_Q1)
            loss_Q1.requires_grad_(True)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(emb_b, emb_s)
            c2_loss = self.critic_2.criterion(emb_b, emb_s, reward)
            loss_Q2 = torch.mean(F.mse_loss(current_Q2, target_Q) + c2_loss)
            loss_Q2.requires_grad_(True)
            # print("loss_Q2", loss_Q2)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            time_end = time.time()
            # print("检查计算出的数值是否符合要求: reward: , ", reward,
            #       "target_q: ", target_Q, ", q1: , q2: ,", current_Q1, current_Q2,
            #       " c1_loss(更新order): ， lossq1: ", c1_loss, loss_Q1)
            # print("计算q1,q2耗时：", time_end- time_begin)
            time_begin = time_end
            # Delayed policy updates:
            if i % args['policy_delay'] == 0:
                # Compute actor loss:
                actor_loss = self.critic_1(emb_b, emb_s)
                actor_loss = - actor_loss.mean()   #希望reward越大越好
                actor_loss.requires_grad_(True)
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                time_end = time.time()
                # print("actor loss耗时", time_end-time_begin)
                time_begin = time_end
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- args['tau']) * target_param.data) + args['tau'] * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - args['tau']) * target_param.data) + args['tau'] * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - args['tau']) * target_param.data) + args['tau'] * param.data)

                self.num_actor_update_iteration += 1
                time_end = time.time()
                # print("转移权重耗时：", time_end-time_begin)
                time_begin = time_end
            # print("一次update总耗时：", time_end - start_time)
        self.num_critic_update_iteration += 1
        self.num_training += 1


    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        torch.save(self.emb_model.state_dict(), directory+'emb_model.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, dir=None):
        if dir is None:
            dir = directory
        self.actor.load_state_dict(torch.load(dir + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(dir + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(dir + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(dir + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(dir + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(dir + 'critic_2_target.pth'))
        self.emb_model.load_state_dict(torch.load(dir + 'emb_model.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = TD3()
    ep_r = 0

    if args['mode'] == 'test':
        agent.load(load_dir)
        pred_res_list = {}
        one_pred = []
        for i in range(origin_graph.graphNum()):
            for j in range(sub_graph.graphNum()):
                subG, mainG = env.reset(i,j, gt_json)
                if subG is None or len(subG)==0:
                    print("主图和子图没有匹配上的，应该直接返回")
                    continue
                for t in count():
                    action, similar = agent.select_action(subG, mainG)
                    one_pred.append(action)
                    # if t<5:
                    #     print("test action", action)
                    next_subG, next_mainG, reward, done, r1_reward = env.step(i, j, action, subG, mainG)
                    ep_r += reward
                    # env.render()
                    if done or t == args['max_episode']:
                        pred_res_list['B%dS%d'%(i,j)] = one_pred
                        one_pred = []
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        if j % args['print_log'] == 0:
                            vis_graph_match(subG, mainG, "graph%d_%dres"%(i, j))
                            print("画出了图%d 和小图%d 的匹配图"%(i,j))
                        break
                    subG = next_subG
                    mainG = next_mainG

        with open('res/'+args['load']+'.json', 'w') as fin:
            json.dump(pred_res_list, fin)
            print("save res")

    elif args['mode'] == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args['load']:
            agent.load(load_dir)

        print(origin_graph.graphNum())
        for i in range(origin_graph.graphNum()):
            for j in range(sub_graph.graphNum()):
                print("================新的图开始了================", j)
                subG, mainG = env.reset(i,j) # shape of state = n*(n+num_label)这需要确定下来n和num_label，先定n=10, num_label = 5
                if len(subG) == 0:
                    print("主图和子图没有匹配上的，应该直接返回")
                    continue
                for t in range(args['max_episode']):
                    action, similar = agent.select_action(subG, mainG)
                    np.set_printoptions(suppress=True, precision=3)
                    if t%10 == 0:
                        print("this action", action, subG.nodes(), subG.mask)

                    next_subG, next_mainG, reward, done, r1_reward = env.step(i, j,
                                            action, subG, mainG)
                    ep_r += reward
                    agent.memory.push((subG, mainG, next_subG, next_mainG, action, reward, np.float16(done)))
                    if j+1 % 10 == 0:
                        print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                    if len(agent.memory.storage) >= args['capacity']-1:
                        agent.update(5)

                    subG = next_subG
                    mainG = next_mainG
                    if done or t == args['max_episode']-1:
                        agent.writer.add_scalar('ep_r', ep_r, global_step=i*100+j)
                        if j % args['print_log'] == 0:
                            vis_graph_match(subG, mainG, "graph%d_%dres"%(i, j))
                            print("Ep_i \t{}-{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, j, ep_r, t))

                        ep_r = 0
                        break #当空循环了2000次就停下来不再继续填图

                if j % args['log_interval'] == 0:
                    print("begin save model")
                    agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
