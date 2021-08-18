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


# Pathwise Derivative Policy Gradient
def get_env():
    env=gym.make('LunarLander-v2')
    return env

class Q_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_net=nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU()
        )
        self.action_net=nn.Sequential(
            nn.Linear(4,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU()
        )
        self.net=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self,state,action):
        hidden=self.state_net(state)+self.action_net(action)
        return self.net(hidden)

    def save(self,now_path):
        check_point=self.state_dict()
        torch.save(check_point,now_path)

    def load(self,now_path):
        data=torch.load(now_path)
        self.load_state_dict(data)

class Policy_Net(nn.Module):
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
        return F.softmax(self.net(state),dim=-1)

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
            

        out_states = torch.stack(out_states).float().detach()
        out_actions = torch.IntTensor(out_actions).float().detach()
        out_rewards = torch.FloatTensor(out_rewards).float().unsqueeze(-1).detach()
        out_next_states = torch.stack(out_next_states).float().detach()
        out_done=torch.BoolTensor(out_done).unsqueeze(-1).detach()

        return out_states,out_actions,out_rewards,out_next_states,out_done



def copy_net(net):
        out=copy.deepcopy(net)
        out.load_state_dict(net.state_dict())
        return out

class QLearning_Agent():
    def __init__(self) -> None:
        self.lr=1e-4
        self.weight_decay=0

        self.train_step=5
        self.batch_size=32

        self.epoch=2000000

        self.writer=None

        self.cache_len=100000

        self.reward_alpha=0.99

        # self.max_paly_time=1000

        self.q_net=None
        self.q_optimizer=None

        self.policy_net=None
        self.policy_optimizer=None

        self.env=get_env()
        self.data=GameData(self.cache_len)
        
        self.train_count=0
        self.c=2
        self.target_q_net=None
        self.target_policy_net=None

    def set_net(self):
        self.q_net=Q_Net()
        self.q_optimizer=optim.Adam(self.q_net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        
        self.policy_net=Policy_Net()
        self.policy_optimizer=optim.Adam(self.policy_net.parameters(),lr=self.lr,weight_decay=self.weight_decay)

        
        self.target_q_net=copy_net(self.q_net)
        self.target_policy_net=copy_net(self.policy_net)

    def learn_q(self,states,actions,rewards,next_states,is_done):
        self.target_policy_net.eval()
        self.target_q_net.eval()
        with torch.no_grad():
            next_actions=self.target_policy_net(next_states)
            next_target=self.target_q_net(next_states,next_actions)
            next_target[is_done]*=0
        
        self.q_net.train()
        now_target=self.reward_alpha*next_target+rewards
        now_except=self.q_net(states,actions)
        loss_func=nn.MSELoss()
        loss=loss_func(now_target,now_except)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

# https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/37
# model.train() and model.eval() do not change any behavior of the gradient calculations, but are used to set specific layers like dropout and batchnorm to evaluation mode (dropout won’t drop activations, batchnorm will use running estimates instead of batch statistics).
# After the with torch.no_grad() block was executed, your gradient behavior will be the same as before entering the block.

# so eval also compute gradient
    def learn_policy(self,states,actions,rewards,next_states,is_done):
        self.q_net.eval()
        self.policy_net.train()
        now_action=self.policy_net(states)
        now_value=self.q_net(states,now_action)

        loss=-now_value.mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


    def learn(self):
        now_collate=GameDataCollate()
        now_loader=DataLoader(dataset=self.data,batch_size=self.batch_size,shuffle=True,collate_fn=now_collate)

        for iter,now_data in enumerate(now_loader):

            states,actions,rewards,next_states,is_done=now_data
            self.learn_q(states,actions,rewards,next_states,is_done)
            self.learn_policy(states,actions,rewards,next_states,is_done)
            break
        self.train_count+=1
        if(self.train_count and self.train_count%self.c==0):
            self.target_q_net=copy_net(self.q_net)
            self.target_policy_net=copy_net(self.policy_net)


    def play_once(self,state):
        self.policy_net.eval()
        with torch.no_grad():
            action_p=self.policy_net(torch.FloatTensor(state))
            action_dist=Categorical(action_p)
            action=action_dist.sample().item()
            next_state,reward,is_done,_=self.env.step(action)
            out_action=np.zeros(4)
            out_action[action]=1
        return state,out_action,reward,next_state,is_done

    def play(self):
        state=self.env.reset()

        total_reward=0
        while True:
            _,_,reward,next_state,is_done=self.play_once(state)
            state=next_state
            total_reward+=reward
            if is_done:
                break
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
                    
                    if i%100==0:
                        print(i,now_total_reward,reward)
                    
                    now_total_reward=0
                    break

                play_step+=1


if __name__ == "__main__":
    # a=QLearning_Agent()
    # a=Double_QLearning_Agent()
    # a.train()
    d_net=Q_Net()
    a=QLearning_Agent()
    a.set_net()
    a.train()