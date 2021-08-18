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
        self.done=deque([],maxlen=max_len)


    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.rewards[index], self.next_states[index],self.done[index]

    def __len__(self):
        return len(self.actions)



class GameDataCollate:
    def __call__(self, batch):
        out_states = []
        out_actions = []
        out_rewards = []
        out_next_states = []
        out_done=[]

        for i in batch:
            tmp_states, tmp_actions, tmp_rewards, tmp_next_states, tmp_done = i
            out_states.append(torch.from_numpy(tmp_states))
            out_actions.append(tmp_actions)
            out_rewards.append(tmp_rewards)
            out_next_states.append(torch.from_numpy(tmp_next_states))
            out_done.append(tmp_done)
            

        out_states = torch.stack(out_states).detach()
        out_actions = torch.IntTensor(out_actions).unsqueeze(-1).detach()
        out_rewards = torch.FloatTensor(out_rewards).unsqueeze(-1).detach()
        out_next_states = torch.stack(out_next_states).detach()
        out_done=torch.BoolTensor(out_done).unsqueeze(-1).detach()

        return out_states,out_actions,out_rewards,out_next_states,out_done



class QLearning_Agent():
    def __init__(self) -> None:
        self.lr=1e-4
        self.weight_decay=0

        self.train_step=5
        self.batch_size=32

        self.epoch=2000000

        self.random_p=0.9
        self.random_p_degree_per=0.9
        self.random_p_degree_step=100
        self.random_p_min=0.1

        self.eval_num=20
        self.eval_step=100

        self.writer=None
        self.target_net=None

        self.cache_len=100000

        self.reward_alpha=0.99

        # self.max_paly_time=1000

        self.net=None
        self.optimizer=None

        self.env=get_env()
        self.data=GameData(self.cache_len)
        self.eval_iter=0

    def set_net(self,net):
        self.net=net
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        self.copy_to_target_net()

    def copy_to_target_net(self):
        if self.target_net is None:
            self.target_net=copy.deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())

    def get_except(self,rewards,next_states,is_done):
        self.target_net.eval()
        self.net.eval()
        with torch.no_grad():
            next_action=torch.argmax(self.net(next_states),dim=-1).unsqueeze(-1)
            next_reward=self.target_net(next_states).gather(1,next_action.long())
            next_reward[is_done]=0
            except_reward=rewards+self.reward_alpha*next_reward
        # self.target_net.eval()
        # with torch.no_grad():
        #     next_reward=torch.max(self.target_net(next_states),dim=-1)[0].unsqueeze(-1)
        #     next_reward[is_done]=0
        #     except_reward=rewards+self.reward_alpha*next_reward
        return except_reward

    def get_estimate(self,states,actions):
        self.net.train()
        now_rewards=self.net(states).gather(1,actions.long())
        return now_rewards

    def learn(self):
        now_collate=GameDataCollate()
        now_loader=DataLoader(dataset=self.data,batch_size=self.batch_size,shuffle=True,collate_fn=now_collate)

        self.copy_to_target_net()
        for iter,now_data in enumerate(now_loader):

            states,actions,rewards,next_states,is_done=now_data
            now_except=self.get_except(rewards,next_states,is_done)
            now_estimate=self.get_estimate(states,actions)

            
            loss_func=nn.MSELoss()
            loss=loss_func(now_estimate,now_except)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()
            break

    def play_once(self,state):
        self.net.eval()
        with torch.no_grad():
            q_value=self.net(torch.FloatTensor(state))
            action=torch.argmax(q_value).item()
            if torch.rand(1).item()<self.random_p:
                action=random.randint(0,3)
            
            next_state,reward,is_done,_=self.env.step(action)
        return state,action,reward,next_state,is_done

    def play(self):
        state=self.env.reset()

        total_reward=0
        while True:
            _,_,reward,next_state,is_done=self.play_once(state)
            state=next_state
            total_reward+=reward
            if is_done:
                break
        self.eval_iter+=1
        return total_reward,reward


    def train(self):
        self.writer = SummaryWriter(os.path.join('logData', time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

        now_total_reward=0
        for i in range(self.epoch):
            state=self.env.reset()
            play_step=0
            while True:
                state,action,reward,next_state,is_done=self.play_once(state)
                self.data.states.append(state)
                self.data.actions.append(action)
                self.data.rewards.append(reward)
                self.data.next_states.append(next_state)
                self.data.done.append(is_done)
                
                state=next_state
                now_total_reward+=reward
                
                if len(self.data)>self.batch_size and play_step%self.train_step==0:
                    self.learn()
                if is_done:
                    self.writer.add_scalar('total',now_total_reward,i)
                    self.writer.add_scalar('final',reward,i)
                    self.writer.add_scalar('data_len',len(self.data),i)
                    self.writer.add_scalar('p',self.random_p,i)
                    
                    now_total_reward=0
                    break

                # if play_step>self.max_paly_time:
                #     now_total_reward=0
                #     break
                
                play_step+=1
            

            if i!=0 and i%self.random_p_degree_step==0:
                self.random_p*=self.random_p_degree_per
                self.random_p=max(self.random_p_min,self.random_p)
            
            
            if i!=0 and i%self.eval_step==0:
                print(i)
                self.eval()
            
    
    def eval(self):
        ori_p=self.random_p
        totals=[]
        fins=[]
        
        for i in range(self.eval_num):
            now_total,now_fin=self.play()
            totals.append(now_total)
            fins.append(now_fin)
        ave_total=np.array(now_total).mean()
        ave_fin=np.array(fins).mean()
        print('eval',ave_total,ave_fin)


        self.random_p=ori_p


if __name__ == "__main__":
    # a=QLearning_Agent()
    # a=Double_QLearning_Agent()
    # a.train()
    d_net=Q_Net()
    a=QLearning_Agent()
    a.set_net(d_net)
    a.train()