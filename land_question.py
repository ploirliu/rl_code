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


def get_env():
    env=gym.make('LunarLander-v2')
    return env

class LayerNet(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU()
            # nn.Linear(hidden_dim,input_dim),
            # nn.ReLU(),
        )
    def forward(self,input_data):
        out=self.net(input_data)
        # out+=input_data
        return out

class RL_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            LayerNet(8,128),
            LayerNet(128,16),
            LayerNet(16,128),
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


def compute_reward(now_reward,alpha):
    now_len=len(now_reward)
    out=[i for i in now_reward]
    for i in reversed(range(1,now_len)):
        out[i-1]+=(alpha*out[i])
    return out

class RL_Agent():
    def __init__(self,net,lr=1e-3,weight_decay=0) -> None:
        self.net=net
        self.optimizer=optim.Adam(self.net.parameters(),lr=lr,weight_decay=weight_decay)
    
    @abstractmethod
    def compute_loss(self,states,actions,probs,rewards):
        pass

    @abstractmethod
    def learn(self,states,actions,probs,rewards):
        self.net.train()
        self.optimizer.zero_grad()

        states=states.detach()
        actions=actions.detach()
        rewards=rewards.detach()
        probs=probs.detach()

        loss=self.compute_loss(states,actions,probs,rewards)

        loss.backward()
        self.optimizer.step()

    def play(self,env,reward_alpha):
        self.net.eval()
        
        state=env.reset()

        now_states=[]
        now_actions=[]
        now_probs=[]
        now_rewards=[]

        total_reward=0
        while True:
            action_prob=self.net(torch.FloatTensor(state))
            action_dist=Categorical(action_prob)
            action=action_dist.sample().item()
            prob=action_prob[action]
            
            now_states.append(state)
            now_actions.append(action)
            now_probs.append(prob)

            state,reward,done,_=env.step(action)
            
            now_rewards.append(reward)

            total_reward+=reward

            if done:
                now_rewards=compute_reward(now_rewards,reward_alpha)
                break
        return now_states,now_actions,now_probs,now_rewards,total_reward,reward

class PolicyGradientAgent(RL_Agent):
    def __init__(self, net, lr, weight_decay) -> None:
        super().__init__(net, lr=lr, weight_decay=weight_decay)

    @abstractmethod
    def compute_loss(self,states,actions,probs,rewards):
        now_action_prob=self.net(states)

        now_probs=[now_action_prob[i,actions[i,0].int()] for i in range(actions.shape[0])]
        now_probs=torch.stack(now_probs).unsqueeze(-1)

        now_log_probs=torch.log(now_probs)
        loss=(-now_log_probs*rewards).mean()
        return loss
    
    
    @abstractmethod
    def learn(self,states,actions,probs,rewards):
        super().learn(states,actions,probs,rewards)

class PPO2_Agent(RL_Agent):
    def __init__(self, net,train_step, lr, weight_decay) -> None:
        super().__init__(net, lr=lr, weight_decay=weight_decay)
        self.train_step=train_step
        self.e=0.2
    
    @abstractmethod
    def compute_loss(self,states,actions,probs,rewards):
        now_action_prob=self.net(states)

        now_probs=[now_action_prob[i,actions[i,0].int()] for i in range(actions.shape[0])]
        now_probs=torch.stack(now_probs).unsqueeze(-1)

        ori_expectation=now_probs*rewards/probs
        clip_expectation=torch.clip(now_probs/probs,1-self.e,1+self.e)*rewards

        fin_expectation=torch.min(ori_expectation,clip_expectation)
        loss=(-fin_expectation).mean()
        return loss
    
    @abstractmethod
    def learn(self,states,actions,probs,rewards):
        for _ in range(self.train_step):
            super().learn(states,actions,probs,rewards)

def AgentFactory(agent_name,lr,train_step):
    net=RL_Net()

    weight_decay=1e-4

    if agent_name=="PPO2":
        out=PPO2_Agent(net,train_step,lr,weight_decay)
    else:
        out=PolicyGradientAgent(net,lr,weight_decay)
    return out


def train(out_dir,agent_name):
    lr=1e-3
    train_step=20

    PLAY_PER_BATCH=10
    NUM_BATCH=100000
    SAVE_BATCH=100
    alpha=0.6

    if os.path.exists(out_dir)==False:
        os.mkdir(out_dir)

    writer = SummaryWriter(os.path.join('logData', time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))+agent_name))
   
    agent=AgentFactory(agent_name,lr,train_step)

    

    env=get_env()

    for iter in range(NUM_BATCH):

        all_total_reward=[]
        all_fin_reward=[]

        all_states,all_actions,all_probs,all_rewards=[],[],[],[]

        for _ in range(PLAY_PER_BATCH):
            now_states,now_actions,now_probs,now_rewards,total_reward,fin_reward=agent.play(env,alpha)

            all_total_reward.append(total_reward)
            all_fin_reward.append(fin_reward)

            all_states+=now_states
            all_actions+=now_actions
            all_probs+=now_probs
            all_rewards+=now_rewards

        ave_total_reward=sum(all_total_reward)/len(all_total_reward)
        ave_fin_reward=sum(all_fin_reward)/len(all_fin_reward)

        if iter%SAVE_BATCH==0:
            print(f"Total:{ave_total_reward:4.1f},Final:{ave_fin_reward:4.1f}")
            agent.net.save(os.path.join(out_dir,str(iter)+'.cpt'))

        writer.add_scalar('total',ave_total_reward,iter)
        writer.add_scalar('final',ave_fin_reward,iter)

        all_rewards=np.array(all_rewards)
        all_rewards=(all_rewards-np.mean(all_rewards))/(np.std(all_rewards)+1e-5)
        # all_rewards=(all_rewards-np.mean(all_rewards))

        all_probs=(torch.stack(all_probs)).unsqueeze(-1)
        all_rewards=(torch.from_numpy(all_rewards)).unsqueeze(-1)
        all_states=torch.from_numpy(np.stack(all_states))
        all_actions=(torch.FloatTensor(all_actions)).unsqueeze(-1)
            
        agent.learn(all_states,all_actions,all_probs,all_rewards)

if __name__ == "__main__":
    train(r'D:\code\rl_code\out','PG')
    # train(r'D:\code\rl_code\out','PPO2')