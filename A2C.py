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


def compute_reward(now_reward,alpha):
    now_len=len(now_reward)
    out=[i for i in now_reward]
    for i in reversed(range(1,now_len)):
        out[i-1]+=(alpha*out[i])
    return out

class A2C_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_net=nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.policy_net=nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )
    
    def forward(self,state):
        policy_out=F.softmax(self.policy_net(state),dim=-1)
        value_out=self.value_net(state)
        return policy_out,value_out

    def save(self,now_path):
        check_point=self.state_dict()
        torch.save(check_point,now_path)

    def load(self,now_path):
        data=torch.load(now_path)
        self.load_state_dict(data)




def get_batch_data(batch):
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


def copy_net(net):
        out=copy.deepcopy(net)
        out.load_state_dict(net.state_dict())
        return out

class TD_Agent():
    def __init__(self) -> None:
        self.lr=1e-4
        self.weight_decay=0

        self.train_step=5
        self.batch_size=32

        self.epoch=2000000

        self.eval_num=20
        self.eval_step=100

        self.writer=None

        self.cache_len=100000

        self.reward_alpha=0.99


        self.net=None
        self.optimizer=None

        self.env=get_env()
        self.data=[]
        self.eval_iter=0

        self.show_epoch=100

    def set_net(self,a2c_net):
        self.net=a2c_net
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        

    @abstractmethod
    def learn_value(self):
        states,_,rewards,_,_=get_batch_data(self.data)

        self.net.train()
        _,now_value=self.net(states)
        loss_func=nn.MSELoss()

        loss=loss_func(now_value,rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    @abstractmethod
    def learn_policy(self):
        states,actions,rewards,next_states,is_done=get_batch_data(self.data)

        self.net.train()
        now_action,now_value=self.net(states)
        now_action_p=now_action.gather(1,actions.long())
        now_action_log=torch.log(now_action_p)
        
        loss=(rewards-now_value)*now_action_log
        loss=(-loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.optimizer.step()
    
    def learn(self):
        self.learn_value()
        self.learn_policy()


    def play_once(self,state):
        self.net.eval()
        with torch.no_grad():
            action_p,_=self.net(torch.FloatTensor(state))
            action_dist=Categorical(action_p)
            action=action_dist.sample().item()
            next_state,reward,is_done,_=self.env.step(action)
        return state,action,reward,next_state,is_done


    
    @abstractmethod
    def train(self):
        self.writer = SummaryWriter(os.path.join('logData', time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

        for i in range(self.epoch):
            now_total_reward=0
            state=self.env.reset()
            play_step=0
            all_rewards=[]
            while True:
                state,action,reward,next_state,is_done=self.play_once(state)
                self.data.append([state,action,reward,next_state,is_done])
                all_rewards.append(reward)
                
                state=next_state
                now_total_reward+=reward
                
                # if play_step and play_step%self.train_step==0:
                #     self.learn()
                if is_done:
                    self.writer.add_scalar('total',now_total_reward,i)
                    self.writer.add_scalar('final',reward,i)

                    
                    all_rewards=compute_reward(all_rewards,self.reward_alpha)
                    for j in range(len(all_rewards)):
                        self.data[j][2]=all_rewards[j]

                    self.learn()
                    self.data=[]
                    
                    break

                play_step+=1


            if i%self.show_epoch==0:
                print(i,now_total_reward,reward)

# class MC_Agent(TD_Agent):
#     def __init__(self) -> None:
#         super().__init__()
    
#     @abstractmethod
#     def learn_value(self):
#         states,actions,rewards,next_states,is_done=get_batch_data(self.data)

#         self.value_net.train()
#         now_estimate=self.value_net(states)
#         next_estimate=self.value_net(next_states)
#         next_estimate[is_done]*=0
#         now_expect=rewards+self.reward_alpha*next_estimate

#         loss_func=nn.MSELoss()

#         loss=loss_func(now_estimate,now_expect)
#         self.value_optimizer.zero_grad()
#         loss.backward()
#         self.value_optimizer.step()
    
#     @abstractmethod
#     def learn_policy(self):
#         states,actions,rewards,next_states,is_done=get_batch_data(self.data)
#         self.value_net.eval()
#         with torch.no_grad():
#             now_value=self.value_net(states)
#             next_value=self.value_net(next_states)

#         self.policy_net.train()
#         now_action=self.policy_net(states)
#         now_action_p=now_action.gather(1,actions.long())
#         now_action_log=torch.log(now_action_p)
        
#         loss=(rewards+self.reward_alpha*next_value-now_value)*now_action_log
#         loss=(-loss).mean()
#         self.policy_optimizer.zero_grad()
#         loss.backward()
#         # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
#         self.policy_optimizer.step()
    
    
#     @abstractmethod
#     def train(self):
#         self.writer = SummaryWriter(os.path.join('logData', time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

#         for i in range(self.epoch):
#             now_total_reward=0
#             state=self.env.reset()
#             while True:
#                 state,action,reward,next_state,is_done=self.play_once(state)
#                 self.data.append([state,action,reward,next_state,is_done])
                
#                 state=next_state
#                 now_total_reward+=reward
                
#                 if is_done:
#                     self.writer.add_scalar('total',now_total_reward,i)
#                     self.writer.add_scalar('final',reward,i)

#                     self.learn()
#                     self.data=[]
                    
#                     break



#             if i%self.show_epoch==0:
#                 print(i,now_total_reward,reward)



# class Double_MC_Agent(MC_Agent):
#     def __init__(self) -> None:
#         super().__init__()
    
#     @abstractmethod
#     def learn_value(self):
#         target_net=copy.deepcopy(self.value_net)
#         target_net.load_state_dict(self.value_net.state_dict())

#         states,actions,rewards,next_states,is_done=get_batch_data(self.data)

#         target_net.eval()
#         with torch.no_grad():
#             next_except=target_net(next_states)
#             next_except[is_done]*=0
#             now_expect=rewards+self.reward_alpha*next_except


#         self.value_net.train()
#         now_estimate=self.value_net(states)

#         loss_func=nn.MSELoss()

#         loss=loss_func(now_estimate,now_expect)
#         self.value_optimizer.zero_grad()
#         loss.backward()
#         self.value_optimizer.step()
    


if __name__ == "__main__":
    a2c_net=A2C_Net()
    a=TD_Agent()
    a.set_net(a2c_net)
    a.train()