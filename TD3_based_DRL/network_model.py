import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 128

def fanin_init(size,fanin=None):
    fanin = fanin or size[0]
    v = 1./ np.sqrt(fanin)
    
    return torch.Tensor(size).uniform_(-v,v)
def he_init(tensor):
    torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')    

class Actor(nn.Module):
    def __init__(self,nb_states,nb_actions,hidden=32,init_w=3e-1):
        super(Actor,self).__init__()
        self.conv1 = nn.Conv2d(1,6,6)
        self.conv2 = nn.Conv2d(6,16,6)
        self.fc1 = nn.Linear(16*16*7,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,nb_actions)
        self.relu = nn.ReLU()
        self.sig = nn.Tanh()
        self.init_weights(init_w)
        
    def init_weights(self,init_w):
        he_init( self.fc1.weight.data )
        he_init( self.fc2.weight.data )
        he_init( self.fc3.weight.data )

        self.fc4.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):
        
        x = inp.unsqueeze(1)

        x = F.max_pool2d( self.conv1(x),2)
        x = F.max_pool2d( self.conv2(x),2)
        x = x.view(x.size(0),16*16*7)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        out = self.sig(x)
        return out
    
class Critic(nn.Module):
    def __init__(self,nb_states,nb_actions,hidden=256,init_w=3e-1):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(nb_states,hidden)
        self.fc2 = nn.Linear(hidden+nb_actions,hidden)
        self.fc3 = nn.Linear(hidden,hidden)
        self.fc4 = nn.Linear(hidden,hidden)
        self.fc5 = nn.Linear(hidden,1)
        
        self.fc11 = nn.Linear(nb_states,hidden)
        self.fc21 = nn.Linear(hidden+nb_actions,hidden)
        self.fc31 = nn.Linear(hidden,hidden)
        self.fc41 = nn.Linear(hidden,hidden)
        self.fc51 = nn.Linear(hidden,1)
        
        self.relu = nn.ReLU()
        self.sig = nn.Tanh()
        self.init_weights(init_w)
        
    def init_weights(self,init_w):
        he_init( self.fc1.weight.data )
        he_init( self.fc2.weight.data )
        he_init( self.fc3.weight.data )
        he_init( self.fc4.weight.data )
        self.fc5.weight.data.uniform_(-init_w,init_w)
        
        he_init( self.fc11.weight.data )
        he_init( self.fc21.weight.data )
        he_init( self.fc31.weight.data )
        he_init( self.fc41.weight.data )
        self.fc51.weight.data.uniform_(-init_w,init_w)
        
    def forward(self,inp):
        x,a = inp
        x = x.view(x.size(0),45*80)
        q1 = self.fc1(x)
        q1 = self.relu(q1)
        q1 = self.fc2(torch.cat([q1,a],1))
        q1 = self.relu(q1)
        q1 = self.fc3(q1)
        q1 = self.relu(q1)
        q1 = self.fc4(q1)
        q1 = self.relu(q1)
        q1 = self.fc5(q1)
        
        q2 = self.fc11(x)
        q2 = self.relu(q2)
        q2 = self.fc21(torch.cat([q2,a],1))
        q2 = self.relu(q2)
        q2 = self.fc31(q2)
        q2 = self.relu(q2)
        q2 = self.fc41(q2)
        q2 = self.relu(q2)
        q2 = self.fc51(q2)
        
        return q1,q2


    
    
    
    