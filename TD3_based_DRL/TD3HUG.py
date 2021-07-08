import pickle
import numpy as np

import torch
import torch.nn as nn

from TD3_based_DRL.priority_replay import Memory
from TD3_based_DRL.network_model import Actor,Critic
from TD3_based_DRL.util import hard_update, soft_update

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Hyperparameters
MEMORY_CAPACITY = 38400
BATCH_SIZE = 128
GAMMA = 0.95
LR_C = 0.0005
LR_A = 0.0002
TAU = 0.001
POLICY_NOSIE = 0.2
POLICY_FREQ = 1
NOISE_CLIP = 0.5

class DRL:
        
    def __init__(self, action_dim, state_dim, LR_C = LR_C, LR_A = LR_A):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Hyperparameters configuration
        self.state_dim = state_dim[0] * state_dim[1]
        self.state_dim_width = state_dim[0]
        self.state_dim_height = state_dim[1]
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOSIE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.itera = 0

        # Priority Experience Replay Buffer
        self.pointer = 0
        self.memory = Memory(MEMORY_CAPACITY)
        
        self.actor = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim,self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LR_A)
        
        self.critic = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_optimizers = torch.optim.Adam(self.critic.parameters(),LR_C)
        
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)
        
            
    def learn(self, batch_size = BATCH_SIZE, epoch=0):

        ## batched state, batched action, batched action from expert, batched intervention signal, batched reward, batched next state
        bs, ba, ba_e, bi, br, bs_, tree_idx, ISweight = self.retrive(batch_size)
        bs = torch.tensor(bs, dtype=torch.float).reshape(batch_size, self.state_dim_height, self.state_dim_width).to(self.device)
        ba = torch.tensor(ba, dtype=torch.float).to(self.device)
        ba_e = torch.tensor(ba_e, dtype=torch.float).to(self.device)
        br = torch.tensor(br, dtype=torch.float).to(self.device)
        bs_ = torch.tensor(bs_, dtype=torch.float).reshape(batch_size, self.state_dim_height, self.state_dim_width).to(self.device)

        # initialize the loss variables
        loss_c, loss_a = 0, 0

        ## calculate the predicted values of the critic
        with torch.no_grad():
            noise1 = (torch.randn_like(ba) * self.policy_noise).clamp(0, 1)
            a_ = (self.actor_target(bs_).detach() + noise1).clamp(0, 1)
            target_q1, target_q2 = self.critic_target([bs_,a_])
            target_q1 = target_q1.detach()
            target_q2 = target_q2.detach()
            target_q = torch.min(target_q1,target_q2)
            y_expected = br + self.gamma * target_q    
        y_predicted1, y_predicted2 = self.critic.forward([bs,ba]) 
        errors = y_expected - y_predicted1

        ## update the critic
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted1,y_expected)+critic_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()
        
        ## update the actor
        if self.itera % self.policy_freq == 0:
            ## distinguish the intervention steps and non-intervention steps, and their corresponded variables
            index1,_ = np.where(bi==0)
            index2,_ = np.where(bi==1)
            bs1,_,_,_=bs[index1],ba[index1],br[index1],bs_[index1]
            bs2,ba2,_,_=bs[index2],ba[index2],br[index2],bs_[index2]
            
            ## if there exists human intervention steps
            if bs2.size(0) != 0:
                # if there exists non-intervention steps
                if bs1.size(0) != 0:
                    bs1 = torch.reshape(bs1,(len(bs1), self.state_dim_height, self.state_dim_width))
                    bs2 = torch.reshape(bs2,(len(bs2), self.state_dim_height, self.state_dim_width))
                    pred_a1 = self.actor.forward(bs1)
                    pred_a2 = self.actor.forward(bs2)
                    # calculate the TD3 loss
                    loss_actor1 = (-self.critic.forward([bs1,pred_a1])[0])
                    
                    with torch.no_grad():
                        # calculate the Q advantage weights
                        weights = torch.exp( self.critic_target([bs2,ba2])[0] - self.critic_target([bs2,pred_a2])[0])
                        weights[weights > 1] = 1
                        weights = weights - 1

                        # calculate the final weights
                        lam = 0.997 ** epoch # lambda is the soft update coefficient
                        weights = lam * 3 + (1-lam) * (torch.exp(weights)) * 3
                    
                    # calculate the human-guidance loss
                    loss_actor2 = weights * ((pred_a2 - ba2)**2)

                    # calculate the final loss
                    loss_actor = torch.cat((loss_actor1,loss_actor2),0).mean()
                
                # if there are all intervention steps
                else:
                    pred_a = self.actor.forward(bs)
                    with torch.no_grad():
                        # calculate the Q advantage weights
                        weights = torch.exp( self.critic_target([bs,ba])[0] - self.critic_target([bs,pred_a])[0])
                        weights[weights > 1] = 1
                        weights = weights - 1

                        # calculate the final weights
                        lam = 0.997 ** epoch # lambda is the soft update coefficient
                        weights = lam * 3 + (1-lam) * (torch.exp(weights)) * 3
                    
                    # calculate the final loss
                    loss_actor = weights * ((pred_a - ba)**2)
                    loss_actor = loss_actor.mean()
            
            ## if there does not exist human intervention steps
            else:
                pred_a = self.actor.forward(bs)
                # calculate the TD3 loss
                loss_actor = (-self.critic.forward([bs,pred_a])[0]).mean()
            
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            #soft replacement for target networks
            soft_update(self.actor_target,self.actor,self.tau)
            soft_update(self.critic_target,self.critic,self.tau)

            loss_a = loss_actor.mean().item()

        loss_c = loss_critic.mean().item()
        
        self.itera += 1

        self.memory.batch_update(tree_idx, abs(errors.detach().cpu().numpy()) )
        
        return loss_c, loss_a


    ## pre-initialization trick, train the critic before the formal learning process of DRL
    def pre_init_critic(self, batch_size=BATCH_SIZE):
        
        ## batched state, batched action, batched action from expert, batched intervention signal, batched reward, batched next state
        bs, ba, ba_e, bi, br, bs_, tree_idx, ISweight = self.retrive(self.batch_size)
        bs = torch.tensor(bs, dtype=torch.float).reshape(batch_size, self.state_dim_height, self.state_dim_width).to(self.device)
        ba = torch.tensor(ba, dtype=torch.float).to(self.device)
        ba_e = torch.tensor(ba_e, dtype=torch.float).to(self.device)
        br = torch.tensor(br, dtype=torch.float).to(self.device)
        bs_ = torch.tensor(bs_, dtype=torch.float).reshape(batch_size, self.state_dim_height, self.state_dim_width).to(self.device)
        
        ## calculate the predicted values of the critic
        with torch.no_grad():
            noise = (torch.randn_like(ba) * self.policy_noise).clamp(0, 1)
            a_ = (self.actor_target(bs_).detach() + noise).clamp(0, 1)
            target_q1, target_q2 = self.critic_target([bs_,a_])
            target_q1 = target_q1.detach()
            target_q2 = target_q2.detach()
            target_q = torch.min(target_q1,target_q2)
            y_expected = br + self.gamma * target_q   
        y_predicted1, y_predicted2 = self.critic.forward([bs,ba])  
        errors = y_expected - y_predicted1

        ## update the critic
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted1,y_expected)+critic_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()

        self.memory.batch_update(tree_idx, abs(errors.detach().cpu().numpy()))

    ## pre-initialization trick, train the actor before the formal learning process of DRL
    def pre_init_actor(self, batch_size=BATCH_SIZE):
        
        ## batched state, batched action
        bs, ba, _, _, _ , _= self.retrive(self.batch_size)
        bs = torch.tensor(bs, dtype=torch.float).reshape(batch_size, self.state_dim_height, self.state_dim_width).to(self.device)
        ba = torch.tensor(ba, dtype=torch.float).to(self.device)

        pred_a = self.actor.forward(bs)
        
        # calculate the supervised learning loss
        loss_actor = ((pred_a - ba)**2).mean()

        # update the actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        

    def choose_action(self,state):
        state = torch.tensor(state, dtype=torch.float).reshape(self.state_dim_height, self.state_dim_width).to(self.device)
        state = state.unsqueeze(0)
        
        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action,-1,1)

        return action
    

    def store_transition(self, s, a, a_e, i, r, s_):

        ## state, action, action from expert, intervention signal, reward, next state
        transition = np.hstack((s, a, a_e, i, r, s_)) 
        self.memory.store(transition)
        self.pointer += 1
    

    def retrive(self, batch_size):

        tree_index, bt, ISWeight = self.memory.sample(batch_size)
        bs = bt[:, :self.state_dim]
        ba = bt[:, self.state_dim: self.state_dim + self.action_dim]
        ba_e = bt[:, self.state_dim + self.action_dim: self.state_dim + self.action_dim + self.action_dim]
        bi = bt[:, -self.state_dim - 2: -self.state_dim - 1]
        br = bt[:, -self.state_dim - 1: -self.state_dim]
        bs_ = bt[:, -self.state_dim:]
        ## batched state, batched action, batched action from expert, batched intervention signal, batched reward, batched next state
        return bs, ba, ba_e, bi, br, bs_, tree_index, ISWeight
    

    def memory_save(self):
        
        per = open("memory.pkl", 'wb')
        str = pickle.dumps(self.memory)
        per.write(str)
        per.close()
    

    def memory_load(self):
        
        with open("memory.pkl",'rb') as file:
            self.memory  = pickle.loads(file.read())
    

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
    
