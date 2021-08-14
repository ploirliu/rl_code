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

from abc import abstractmethod

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
import random

from collections import deque

import copy

def get_env():
    env=gym.make('LunarLander-v2')
    return env

class Q_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )
    
    def forward(self,state):
        return self.net(state)

    def save(self,now_path):
        check_point=self.state_dict()
        torch.save(check_point,now_path)

    def load(self,now_path):
        data=torch.load(now_path)
        self.load_state_dict(data)



class GameData(Dataset):
    def __init__(self,max_len):
        self.states=deque([],maxlen=max_len)
        self.actions=deque([],maxlen=max_len)
        self.rewards=deque([],maxlen=max_len)
        self.next_states=deque([],maxlen=max_len)


    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.rewards[index], self.next_states[index]

    def __len__(self):
        return len(self.actions)



class GameDataCollate:
    def __call__(self, batch):
        out_states = []
        out_actions = []
        out_rewards = []
        out_next_states = []
        for i in batch:
            tmp_states, tmp_actions, tmp_rewards, tmp_next_states = i
            out_states.append(torch.from_numpy(tmp_states))
            out_actions.append(tmp_actions)
            out_rewards.append(tmp_rewards)
            out_next_states.append(torch.from_numpy(tmp_next_states))
            

        out_states = torch.stack(out_states).detach()
        out_actions = torch.IntTensor(out_actions).unsqueeze(-1).detach()
        out_rewards = torch.FloatTensor(out_rewards).unsqueeze(-1).detach()
        out_next_states = torch.stack(out_next_states).detach()

        return out_states,out_actions,out_rewards,out_next_states



class QLearning_Agent():
    def __init__(self) -> None:
        self.lr=1e-4
        self.weight_decay=1e-5

        self.train_step=50
        self.batch_size=200

        self.epoch=2000000
        self.show_epoch=50

        self.random_p=0.3
        self.random_p_degree=0.05
        self.random_p_degree_step=200
        self.random_p_min=0.05

        self.eval_num=10

        self.writer=None

        self.c_step=20

        self.copy_net=None

        self.play_num=2

        self.cache_len=30000

        self.min_total_reward=-2000

        # self.net=Q_Net()
        # self.optimizer=optim.Adam(self.net.parameters(),lr=lr,weight_decay=weight_decay)

        self.net=None
        self.optimizer=None

        self.env=get_env()

        # self.copy_net=None
        # self.get_copy_net()
        self.data=GameData(self.cache_len)

    def set_net(self,net):
        # self.net=Q_Net()
        self.net=net
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)

    def get_copy_net(self):
        if self.copy_net is None:
            self.copy_net=copy.deepcopy(self.net)
        self.copy_net.load_state_dict(self.net.state_dict())
        return self.copy_net
    
    @abstractmethod
    def compute_loss(self,states,actions,rewards,next_states):
        self.copy_net.eval()
        next_max_rewards,_= torch.max(self.copy_net(next_states),dim=-1)
        next_max_rewards=next_max_rewards.unsqueeze(-1)

        except_reward=next_max_rewards+rewards

        now_select_rewards=self.net(states).gather(1,actions.long())
        
        loss_func=nn.SmoothL1Loss()
        loss=loss_func(now_select_rewards,except_reward)
        return loss

        

    @abstractmethod
    def learn(self):
        now_collate=GameDataCollate()
        now_loader=DataLoader(dataset=self.data,batch_size=self.batch_size,shuffle=True,collate_fn=now_collate)

        for iter,now_data in enumerate(now_loader):
            if iter>self.train_step:
                break
            
            if iter%self.c_step==0:
                self.copy_net=self.get_copy_net()

            states,actions,rewards,next_states=now_data
            self.net.train()
            self.optimizer.zero_grad()
            loss=self.compute_loss(states,actions,rewards,next_states)

            loss.backward()
            # for p in self.net.parameters():
            #     p.grad.data.clamp_(-1,1)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()
        # self.copy_net=self.get_copy_net()

    def play(self,env):
        self.net.eval()
        
        state=env.reset()

        now_states=[]
        now_actions=[]
        now_rewards=[]
        next_states=[]

        total_reward=0
        while True:
            q_value=self.net(torch.FloatTensor(state))
            action=torch.argmax(q_value).item()
            if torch.rand(1)<self.random_p:
                action=random.randrange(q_value.shape[0])
            
            now_states.append(state)
            now_actions.append(action)
            state,reward,done,_=env.step(action)
            
            now_rewards.append(reward)
            next_states.append(state)

            total_reward+=reward

            if total_reward<self.min_total_reward:
                done=True

            if done:
                break
        for i in range(len(now_states)):
            self.data.states.append(now_states[i])
            self.data.actions.append(now_actions[i])
            self.data.rewards.append(now_rewards[i])
            self.data.next_states.append(next_states[i])
        return total_reward,reward

    def train(self):
        self.writer = SummaryWriter(os.path.join('logData', time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
        for i in range(self.epoch):
            all_total_reward,all_fin_reward=[],[]
            for _ in range(self.play_num):
                total_reward,fin_ward=self.play(self.env)
                all_total_reward.append(total_reward)
                all_fin_reward.append(fin_ward)
            self.writer.add_scalar('total',np.array(all_total_reward).mean(),i)
            self.writer.add_scalar('final',np.array(all_fin_reward).mean(),i)
            if i%self.show_epoch == 0:
                print(i,total_reward,fin_ward,self.random_p)
                self.eval()
            self.learn()

            if i!=0 and i%self.random_p_degree_step==0:
                self.random_p-=self.random_p_degree
                self.random_p=max(self.random_p_min,self.random_p)
    
    def eval(self):
        ori_p=self.random_p
        totals=[]
        fins=[]
        
        for i in range(self.eval_num):
            now_total,now_fin=self.play(self.env)
            totals.append(now_total)
            fins.append(now_fin)
        ave_total=np.array(now_total).mean()
        ave_fin=np.array(fins).mean()
        print('eval',ave_total,ave_fin)


        self.random_p=0
        self.random_p=ori_p

class Double_QLearning_Agent(QLearning_Agent):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute_loss(self, states, actions, rewards, next_states):
        self.copy_net.eval()

        next_all_rewards=self.copy_net(next_states)
        now_next_action=torch.argmax(self.net(next_states),dim=-1).unsqueeze(-1)
        except_next_rewards=next_all_rewards.gather(1,now_next_action.long())
        except_reward=except_next_rewards+rewards
        except_reward.detach()

        now_select_rewards=self.net(states).gather(1,actions.long())
        
        loss_func=nn.SmoothL1Loss()
        loss=loss_func(now_select_rewards,except_reward)
        return loss

class Dueling_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,16),
            nn.ReLU()
        )
        self.v_net=nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.a_net=nn.Sequential(
            nn.Linear(16,64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.LayerNorm(128),
            nn.Linear(128,4,bias=False),
        )
    
    def forward(self,state):
        hidden_data=self.net(state)
        v_value=self.v_net(hidden_data)
        a_value=self.a_net(hidden_data)
        a_value=(a_value.T-a_value.mean(dim=-1)).T
        return a_value+v_value

    def save(self,now_path):
        check_point=self.state_dict()
        torch.save(check_point,now_path)

    def load(self,now_path):
        data=torch.load(now_path)
        self.load_state_dict(data)

class NoiseNet_agent(Double_QLearning_Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_noise_net(self):
        noise_net=copy.deepcopy(self.net)
        all_params=[]
        noise_scale=0.2
        for i in noise_net.parameters():
            all_params.append(i.item())
        now_std=np.array(all_params).std()
        for i in noise_net.paramsters():
            i+=now_std*torch.randn(1)
        return noise_net
        

    @abstractmethod
    def play(self, env):
        self.net.eval()
        
        state=env.reset()

        now_states=[]
        now_actions=[]
        now_rewards=[]
        next_states=[]

        total_reward=0
        while True:
            q_value=self.net(torch.FloatTensor(state))
            action=torch.argmax(q_value).item()
            if torch.rand(1)<self.random_p:
                action=random.randrange(q_value.shape[0])
            
            now_states.append(state)
            now_actions.append(action)
            state,reward,done,_=env.step(action)
            
            now_rewards.append(reward)
            next_states.append(state)

            total_reward+=reward

            if total_reward<self.min_total_reward:
                done=True

            if done:
                break
        for i in range(len(now_states)):
            self.data.states.append(now_states[i])
            self.data.actions.append(now_actions[i])
            self.data.rewards.append(now_rewards[i])
            self.data.next_states.append(next_states[i])
        return total_reward,reward


if __name__ == "__main__":
    # a=QLearning_Agent()
    # a=Double_QLearning_Agent()
    # a.train()
    due_net=Dueling_Net()
    a=Double_QLearning_Agent()
    a.set_net(due_net)
    a.get_copy_net()
    a.train()