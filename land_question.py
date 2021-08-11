import numpy as np
from numpy.core.fromnumeric import clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm

import gym

import matplotlib.pyplot as plt

import os

import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.path.join('logData', time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))


def get_env():
    env=gym.make('LunarLander-v2')
    return env

class PolicyGradientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,4)
        )
    
    def forward(self,state):
        return F.softmax(self.net(state),dim=-1)

    def save(self,now_path):
        check_point=self.state_dict()
        torch.save(check_point,now_path)

    def load(self,now_path):
        data=torch.load(now_path)
        self.load_state_dict(data)


class PolicyGradientAgent():
    def __init__(self,network) -> None:
        self.net=network
        self.optimizer=optim.Adam(self.net.parameters(),lr=1e-3,weight_decay=1e-4)
        # self.optimizer=optim.Adam(self.net.parameters())
    
    def learn(self,log_probs,reward):
        # loss=(-log_probs*reward).sum()
        loss=(-log_probs*reward).mean()
        # loss/=play_num
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self,state):
        action_prob=self.net(torch.FloatTensor(state))
        action_dist=Categorical(action_prob)
        action=action_dist.sample()
        log_prob=action_dist.log_prob(action)
        return action.item(),log_prob


def compute_reward(now_reward,alpha):
    now_len=len(now_reward)
    out=[i for i in now_reward]
    for i in reversed(range(1,now_len)):
        out[i-1]+=(alpha*out[i])
    return out


def train(dir_path):
    network=PolicyGradientNet()
    agent=PolicyGradientAgent(network)

    agent.net.train()
    EPISODE_PER_BATCH=10
    NUM_BATCH=100000

    SAVE_BATCH=100

    avg_total_rewards,avg_fin_rewards=[],[]
    env=get_env()
    alpha=0.6

    for iter in range(NUM_BATCH):
        log_probs,rewards=[],[]
        total_rewards,fin_rewards=[],[]

        for _ in range(EPISODE_PER_BATCH):
            state=env.reset()
            total_reward=0

            now_log_probs,now_reward=[],[]

            while True:
                action,log_prob=agent.sample(state)
                state,reward,done,_=env.step(action)

                now_log_probs.append(log_prob)
                now_reward.append(reward)

                total_reward+=reward

                if done:
                    total_rewards.append(total_reward)
                    fin_rewards.append(reward)

                    now_reward=compute_reward(now_reward,alpha)
                    log_probs=log_probs+now_log_probs
                    rewards=rewards+now_reward
                    break
        ave_total_reward=sum(total_rewards)/len(total_rewards)
        ave_fin_reward=sum(fin_rewards)/len(fin_rewards)
        avg_total_rewards.append(ave_total_reward)
        avg_fin_rewards.append(ave_fin_reward)

        if iter%SAVE_BATCH==0:
            print(f"Total:{ave_total_reward:4.1f},Final:{ave_fin_reward:4.1f}")
            agent.net.save(os.path.join(dir_path,str(iter)+'.cpt'))

        writer.add_scalar('total',ave_total_reward,iter)
        writer.add_scalar('final',ave_fin_reward,iter)

        rewards=np.array(rewards)
        # rewards=(rewards-np.mean(rewards))/(np.std(rewards)+1e-5)
        rewards=(rewards-np.mean(rewards))

        agent.learn(torch.stack(log_probs),torch.from_numpy(rewards))
    # return avg_total_rewards,avg_fin_rewards


class PPO2_Agent():
    def __init__(self,network,train_step,e) -> None:
        self.net=network
        self.optimizer=optim.Adam(self.net.parameters(),lr=1e-3,weight_decay=1e-4)
        # self.optimizer=optim.Adam(self.net.parameters())
        self.train_step=train_step
        self.e=e
    
    def learn(self,state,action,probs,reward):
        self.net.train()
        for _ in range(self.train_step):
            state=state.detach()
            action=action.detach()
            probs=probs.detach()
            reward=reward.detach()


            now_action_prob=self.net(state)

            now_probs=[now_action_prob[i,action[i,0].int()] for i in range(action.shape[0])]
            now_probs=torch.stack(now_probs).unsqueeze(-1)

            ori_expectation=now_probs*reward/probs
            clip_expectation=torch.clip(now_probs/probs,1-self.e,1+self.e)*reward

            fin_expectation=torch.min(ori_expectation,clip_expectation)
            loss=(-fin_expectation).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def sample(self,state):
        action_prob=self.net(torch.FloatTensor(state))
        action_dist=Categorical(action_prob)
        action=action_dist.sample()
        prob=action_prob[action.item()]
        return action.item(),prob

def PPO2_train(dir_path):
    
    EPISODE_PER_BATCH=10
    NUM_BATCH=100000

    SAVE_BATCH=100
    alpha=0.6

    ppo_train_step=3
    ppo_e=0.2

    network=PolicyGradientNet()
    agent=PPO2_Agent(network,ppo_train_step,ppo_e)

    avg_total_rewards,avg_fin_rewards=[],[]
    env=get_env()

    for iter in range(NUM_BATCH):
        
        agent.net.eval()

        states,actions,probs,rewards=[],[],[],[]
        total_rewards,fin_rewards=[],[]

        for _ in range(EPISODE_PER_BATCH):
            state=env.reset()


            total_reward=0

            now_reward=[]

            while True:
                action,prob=agent.sample(state)

                states.append(state)
                actions.append(action)

                state,reward,done,_=env.step(action)

                probs.append(prob)
                now_reward.append(reward)

                total_reward+=reward

                if done:
                    total_rewards.append(total_reward)
                    fin_rewards.append(reward)

                    now_reward=compute_reward(now_reward,alpha)
                    rewards=rewards+now_reward
                    break
        ave_total_reward=sum(total_rewards)/len(total_rewards)
        ave_fin_reward=sum(fin_rewards)/len(fin_rewards)
        avg_total_rewards.append(ave_total_reward)
        avg_fin_rewards.append(ave_fin_reward)

        if iter%SAVE_BATCH==0:
            print(f"Total:{ave_total_reward:4.1f},Final:{ave_fin_reward:4.1f}")
            agent.net.save(os.path.join(dir_path,str(iter)+'.cpt'))

        writer.add_scalar('total',ave_total_reward,iter)
        writer.add_scalar('final',ave_fin_reward,iter)

        rewards=np.array(rewards)
        # rewards=(rewards-np.mean(rewards))/(np.std(rewards)+1e-5)
        rewards=(rewards-np.mean(rewards))

        probs=(torch.stack(probs)).unsqueeze(-1)
        rewards=(torch.from_numpy(rewards)).unsqueeze(-1)
        states=torch.from_numpy(np.stack(states))
        actions=(torch.FloatTensor(actions)).unsqueeze(-1)
        agent.learn(states,actions,probs,rewards)
    # return avg_total_rewards,avg_fin_rewards


if __name__ == "__main__":
    PPO2_train(r'D:\code\rl_code\ppo2_out')
    
    # train(r'D:\code\rl_code\out')

    # avg_total_rewards,avg_fin_rewards=train()

    # plt.plot(avg_total_rewards)
    # plt.title('total reward')
    # plt.show()

    # plt.plot(avg_fin_rewards)
    # plt.title('final reward')
    # plt.show()