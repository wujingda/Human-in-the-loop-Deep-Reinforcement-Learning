# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:52:08 2019

@author: RRC1
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
from TD3_based_DRL.Priority_Replay import Memory
from TD3_based_DRL.network_model import Actor,Critic
from TD3_based_DRL.util import hard_update, soft_update

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



MEMORY_CAPACITY = 38400
BATCH_SIZE = 128
GAMMA = 0.95
LR_C = 0.00055
LR_A = 0.00022
LR_I = 0.01
TAU = 0.001
POLICY_NOSIE = 0.2
POLICY_FREQ = 1
NOISE_CLIP = 0.5

class DRL:
        
    def __init__(self,action_dim,state_dim, LR_C = LR_C, LR_A = LR_A):
        self.use_cuda = True
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOSIE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.itera = 0


        self.pointer = 0
        self.memory = Memory(MEMORY_CAPACITY)   #priority ER
        
        self.actor = Actor(self.state_dim,self.action_dim)
        self.actor_target = Actor(self.state_dim,self.action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LR_A)
        
        self.critic = Critic(self.state_dim,self.action_dim)
        self.critic_target = Critic(self.state_dim,self.action_dim)
        self.critic_optimizers = torch.optim.Adam(self.critic.parameters(),LR_C)
        
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)
        
        if self.use_cuda:
            self.cuda()
            
    def learn(self, batch_size = BATCH_SIZE, epoch=0):
        bs, ba, ba_e, bsup, br, bs_ = self.retrive(batch_size)
        bs = torch.tensor(bs, dtype=torch.float).reshape(batch_size,45,80)
        ba = torch.tensor(ba, dtype=torch.float)
        ba_e = torch.tensor(ba_e, dtype=torch.float)
        br = torch.tensor(br, dtype=torch.float)
        bs_ = torch.tensor(bs_, dtype=torch.float).reshape(batch_size,45,80)
        
        if self.use_cuda:
            bs = bs.cuda()
            ba = ba.cuda()
            ba_e = ba_e.cuda()
            br = br.cuda()
            bs_ = bs_.cuda()
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
        

        loss_a = 0
        #actor gradient


        
        #apply nosie in DDPG
        with torch.no_grad():
            noise1 = (torch.randn_like(ba) * self.policy_noise).clamp(0,1)
            a_ = (self.actor_target(bs_).detach() + noise1).clamp(0,1)
            # a_2 = ba2
            # a_ = torch.cat([a_1,a_2])
            target_q1, target_q2 = self.critic_target([bs_,a_])
            target_q1 = target_q1.detach()
            target_q2 = target_q2.detach()
            target_q = torch.min(target_q1,target_q2)
            y_expected = br + self.gamma * target_q    #priority ER
        y_predicted1, y_predicted2 = self.critic.forward([bs,ba])    #priority ER
        
        #critic gradient
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted1,y_expected)+critic_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()
        
        if self.itera % self.policy_freq == 0:
            index1,_ = np.where(bsup==0)
            index2,_ = np.where(bsup==1)
            bs1,_,_,_=bs[index1],ba[index1],br[index1],bs_[index1]
            bs2,ba2,_,_=bs[index2],ba[index2],br[index2],bs_[index2]
            
            if bs2.size(0) != 0:
                if bs1.size(0) != 0:
                    bs1 = torch.reshape(bs1,(len(bs1),45,80))
                    bs2 = torch.reshape(bs2,(len(bs2),45,80))
                    pred_a1 = self.actor.forward(bs1)
                    pred_a2 = self.actor.forward(bs2)
                    loss_actor1 = (-self.critic.forward([bs1,pred_a1])[0])
                    
                    with torch.no_grad():
                        lam = 0.997**epoch
                        weights0 = self.critic_target([bs2,ba2])[0] - self.critic_target([bs2,pred_a2])[0]
                        weights = lam*4 + (1-lam)*(torch.sigmoid(weights0))*4
                        
                    loss_actor2 = weights*((pred_a2 - ba2)**2)
                    loss_actor = torch.cat((loss_actor1,loss_actor2),0).mean()
                else:
                    # pred_a1 = self.actor.forward(bs2)
                    pred_a = self.actor.forward(bs)
                    # loss_actor1 = (-self.critic.forward([bs2,pred_a1])[0])
                    loss_actor = 3*((pred_a - ba)**2)
                    loss_actor = loss_actor.mean()
            else:
                pred_a = self.actor.forward(bs)
                loss_actor = (-self.critic.forward([bs,pred_a])[0]).mean()
            self.actor_optimizer.zero_grad()
#            loss_actor.backward()
            loss_actor.sum().backward()
            self.actor_optimizer.step()

            

            #soft replacement
            soft_update(self.actor_target,self.actor,self.tau)
            soft_update(self.critic_target,self.critic,self.tau)
            loss_a = loss_actor.sum().item()

        loss_c = loss_critic.mean().item()
        
        self.itera += 1
        
        return loss_c,loss_a

    def pre_init_critic(self):
        
        bs, ba, ba_e, bsup, br, bs_ = self.retrive(self.batch_size)
        bs = torch.tensor(bs, dtype=torch.float).reshape(BATCH_SIZE,45,80)
        ba = torch.tensor(ba, dtype=torch.float)
        ba_e = torch.tensor(ba_e, dtype=torch.float)
        br = torch.tensor(br, dtype=torch.float)
        bs_ = torch.tensor(bs_, dtype=torch.float).reshape(BATCH_SIZE,45,80)
        
        if self.use_cuda:
            bs = bs.cuda()
            ba = ba.cuda()
            ba_e = ba_e.cuda()
            br = br.cuda()
            bs_ = bs_.cuda()
            self.cuda()
        
        with torch.no_grad():
            noise = (torch.randn_like(ba) * self.policy_noise).clamp(0,1)
            a_ = (self.actor_target(bs_).detach() + noise).clamp(0,1)
            target_q1, target_q2 = self.critic_target([bs_,a_])
            target_q1 = target_q1.detach()
            target_q2 = target_q2.detach()
            target_q = torch.min(target_q1,target_q2)
            y_expected = br + self.gamma * target_q    #priority ER
        y_predicted1, y_predicted2 = self.critic.forward([bs,ba])    #priority ER
        
        #critic gradient
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted1,y_expected)+critic_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()
    
    
    def pre_init_actor(self):
        bs, ba, _, _, _ , _= self.retrive(self.batch_size)
        bs = torch.tensor(bs, dtype=torch.float).reshape(BATCH_SIZE,45,80)
        ba = torch.tensor(ba, dtype=torch.float)

        if self.use_cuda:
            bs = bs.cuda()
            ba = ba.cuda()
            self.actor.cuda()
            self.actor_target.cuda()

        pred_a = self.actor.forward(bs)
        
        loss_actor = ((pred_a - ba)**2)
        self.actor_optimizer.zero_grad()

        loss_actor.sum().backward()
        self.actor_optimizer.step()
        
        
    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
                
    def choose_action(self,state):
        state = torch.tensor(state,dtype=torch.float).reshape(45,80)
        state = state.unsqueeze(0)
        
        if self.use_cuda:
            state = state.cuda()
        
        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action,-1,1)
        
        self.action = action

        return action
    
    def store_transition(self, s, a, a_e, sup, r, s_):
        transition = np.hstack((s, a, a_e, sup, r, s_))   #priority ER
        self.memory.store(transition)
        self.pointer += 1
    
    def retrive(self, batch_size):
        tree_index, bt, ISWeight = self.memory.sample(batch_size)    #priority ER
        bs = bt[:, :self.state_dim]
        ba = bt[:, self.state_dim: self.state_dim + self.action_dim]
        ba_e = bt[:, self.state_dim + self.action_dim: self.state_dim + self.action_dim + self.action_dim]
        bsup = bt[:, -self.state_dim - 2: -self.state_dim - 1]
        br = bt[:, -self.state_dim - 1: -self.state_dim]
        bs_ = bt[:, -self.state_dim:]
        
        return bs,ba,ba_e,bsup,br,bs_
    
    def memory_save(self):
        
        per = open("memory_jingda.pkl", 'wb')
        str = pickle.dumps(self.memory)
        per.write(str)
        per.close()
    
    def memory_load(self):
        
        with open("memory_jingda.pkl",'rb') as file:
            self.memory  = pickle.loads(file.read())
        
    def driver(self,state):
        inp = torch.ones(3)
        inp[0] = state[0]
        inp[1] = state[1]
        inp[2] = state[2]
        return self.imitator(inp).detach().cpu().numpy()
    
    def load_model(self, output):
        if output is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
        
    def save(self, log_dir, epoch):
        state = {'actor':self.actor.state_dict(), 'actor_target':self.actor_target.state_dict(),
                 'actor_optimizer':self.actor_optimizer.state_dict(), 
                 'critic':self.critic.state_dict(), 'critic_target':self.critic_target.state_dict(),
                 'critic_optimizers':self.critic_optimizers.state_dict(),
                 'epoch':epoch}
        torch.save(state, log_dir)
        
    def load(self, log_dir):
        checkpoint = torch.load(log_dir)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizers.load_state_dict(checkpoint['critic_optimizers'])
        
    def _sigmode(self, x, sigma = 0.1):
        f = 2 / (1 + np.exp(-sigma * x) ) - 1
        return f
        
        
        
        
        
        
        