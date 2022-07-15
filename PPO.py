import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model import Critic, Actor
from config import args
import datetime
from gcn.config import get_default_config
from env import GNN_env
from dataCenter import gen_Gset
import numpy as np

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print("using device: ", device)
model_args = get_default_config()
f1 = "./data/com-amazon.ungraph.txt"
origin_graph, sub_graph, test_graph = gen_Gset(f1)
env = GNN_env(1, origin_graph, sub_graph)

#############################
# basic setup
max_ep_len = 1000
update_timestep = max_ep_len * 1      # update policy every n timesteps
print_freq = max_ep_len // 10       # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e3)
max_training_timesteps = 3e4
K_epochs = 10               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0
if random_seed:
    torch.manual_seed(random_seed)
    # np.random.seed(random_seed)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.state_s = []
        self.state_b = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.state_s[:]
        del self.state_b[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.emb_model.post_mp.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self):
        print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, subNMNeighbor, gNMNeighbor):

        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob, fea_s, fea_b, action_prob = self.policy_old.act(subNMNeighbor, gNMNeighbor)

        self.buffer.state_s.append(subNMNeighbor)
        self.buffer.state_b.append(gNMNeighbor)
        self.buffer.actions.append(action_prob)
        self.buffer.logprobs.append(action_logprob)

        return action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:  # 这样做是对的，因为第一个reward最重要
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # print(len(self.buffer.actions), len(self.buffer.state_s), len(self.buffer.state_b), len(self.buffer.logprobs))
        # old_state_s = torch.squeeze(torch.stack(self.buffer.state_s, dim=0)).detach().to(device)
        # old_state_b = torch.squeeze(torch.stack(self.buffer.state_b, dim=0)).detach().to(device)
        # old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for ke in range(self.K_epochs):
            # Evaluating old actions and values
            logp_list, state_vlist, dist_list = self.policy.evaluate(
                self.buffer.state_s, self.buffer.state_b, self.buffer.actions)

            state_values = torch.stack(state_vlist)
            logprobs = torch.stack(logp_list)
            dist_entropy = torch.stack(dist_list)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            # 第ke次采样相对于最初的变化有多少
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # 优势函数居然是reward和计算出的q值之间的差距吗
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # loss 分为三个部分，一个是向好的策略方向优化，
            # 一个是督促critic输出准确，还有一个是（姑且理解为）采样的熵
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            print("loss:", torch.sum(loss))
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = Actor(device)
        self.critic = Critic(model_args)

    def act(self, subNMNeighbor, gNMNeighbor):
        _, action_probs, fea_s, fea_b = self.actor(subNMNeighbor, gNMNeighbor)
        action_probs_soft = F.softmax(action_probs.view(-1), dim=-1)

        data_size, _ = fea_b.shape
        dist = Categorical(action_probs_soft)
        action = dist.sample()
        res_action = [0, 0]
        res_action[0] = list(subNMNeighbor.nodes())[action // data_size]
        res_action[1] = list(gNMNeighbor.nodes())[action % data_size]
        # print("get action", action, res_action)
        action_logprob = dist.log_prob(action)

        return res_action, action_logprob.detach(), fea_s, fea_b, action

    def evaluate(self, subNMNeighborL, gNMNeighborL, actionL, batchify=True):
        start_time = datetime.datetime.now()
        _, action_probsL, fea_sL, fea_bL = self.actor(subNMNeighborL,
                                                      gNMNeighborL, batchify=batchify)
        print("embed time: ", datetime.datetime.now()-start_time)
        state_vlist = []
        logp_list = []
        dist_list = []
        for action_probs, fea_s, fea_b, action in zip(
                action_probsL, fea_sL, fea_bL, actionL):
            state_values = self.critic(fea_s, fea_b, action_probs)

            action_probs_soft = F.softmax(action_probs.view(-1), dim=-1)
            dist = Categorical(action_probs_soft)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            # 为什么这个critic只接受state做参数呢
            state_vlist.append(state_values)
            logp_list.append(action_logprobs)
            dist_list.append(dist_entropy)
        print("eval time: ", datetime.datetime.now() - start_time)
        return logp_list, state_vlist, dist_list


def train():
    ppo_agent = PPO(lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    log_f = open('./ppo/log/0712.log', "w+")
    checkpoint_path = "./ppo/ckpt/linshi.ckpt"
    log_f.write('episode,timestep,reward\n')
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    big_g_n, small_g_n = origin_graph.graphNum(), sub_graph.graphNum()
    while time_step <= max_training_timesteps:
        b_i, s_j = (i_episode//small_g_n) % big_g_n, (i_episode) % small_g_n
        subG, mainG = env.reset(b_i, s_j)
        current_ep_reward = 0

        for t in range(1, args['max_episode'] + 1):

            # select action with policy
            # action, similar, _, _ = ppo_agent.select_action(subG, mainG)
            action = ppo_agent.select_action(subG, mainG)
            subG, mainG, reward, done, r1_reward = env.step(b_i, s_j, action, subG, mainG)
            # state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
    log_f.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()