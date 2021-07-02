import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio

import torch
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./TD3_based_DRL/checkpoints/log')

from env import scenario

import os
import pygame
import signal
from utils import signal_handler
from TD3_based_DRL.Priority_Replay import Memory
from reward_shaping import RND



seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





### Arguments
control_freq = 1
MAX_EPISODES = 1500 
LOAD = 0





### Construct the DRL agent
algorithm = 0

if algorithm == 0:
    from TD3_based_DRL.TD3HUG import DRL
    log_dir = 'TD3_based_DRL/checkpoints/TD3jingda.pth'
elif algorithm == 1:
    from TD3_based_DRL.TD3IARL import DRL
    log_dir = 'TD3_based_DRL/checkpoints/TD3IARL.pth'
elif algorithm == 2:
    from TD3_based_DRL.TD3 import DRL   
    log_dir = 'TD3_based_DRL/checkpoints/TD3HIRL.pth'
else:
    from TD3_based_DRL.TD3 import DRL    
    log_dir = 'TD3_based_DRL/checkpoints/TD3.pth'

s_dim = 80*45
a_dim = 1
DRL = DRL(a_dim, s_dim)
var = 0.5

 if PRE_INIT:
     DRL.memory_load()
     for _ in range(10000):
         DRL.pre_init_critic()

if os.path.exists(log_dir) and LOAD:
    checkpoint = torch.load(log_dir)
    DRL.load(log_dir)
    start_epoch = checkpoint['epoch'] + 1




### Scope variables initialization
total_step = 0
a_loss,c_loss = 0,0
start_epoch = 0

loss_critic = []
loss_actor = []

mean_reward_list = []
real_reward_list = []
total_reward_list = []
duration_list =[]

human_record = []
previous_action = [[] for i in range(MAX_EPISODES)]
final_action = [[] for i in range(MAX_EPISODES)]

reward_i_record = [[] for i in range(MAX_EPISODES)]
reward_e_record = [[] for i in range(MAX_EPISODES)]
reward_record = [[] for i in range(MAX_EPISODES)]

start_time = time.perf_counter()

action_disturbing_degree = []
intervene_percent_per_episode = [[] for i in range(MAX_EPISODES)]
intervene_percent = []

x_per_episode = [[] for i in range(MAX_EPISODES)]
y_per_episode = [[] for i in range(MAX_EPISODES)]

qlist = []

from scipy.interpolate import interp1d
waypoint_x_mark = np.array([200,212.5,225,237.5,250,300])
waypoint_y_mark = np.array([335,336.5,338,337.5,335,334])
waypoint = interp1d(waypoint_x_mark, waypoint_y_mark,kind='cubic')

env = scenario(random_spawn=False)

# State, scope = env.obtain_observation()

for i in range(start_epoch, MAX_EPISODES):

    mean_reward = 0
    ep_reward = 0      
    step = 0
    x_i = 0
    v_tar = 10
    done = 0

    flag = 0
    flag1 = []
    flag1.append(None)
    flag_intervene = 0
    flag_qrecord = 0
    
    rand = np.random.randint(0,3)

        
    if i==start_epoch:
        env.destroy()
    
    env = scenario(random_spawn=False)
    State, scope = env.obtain_observation()

    while True:
        
        out_guide = False
        
        if step % control_freq == 0:
            action = DRL.choose_action(State)
            action = np.squeeze( np.clip( np.random.normal(action,var), -1,1) )
        else:
            action = final_action[i][-1] if total_step > 1 else 0.
        
        previous_action[i].append(action)
        
        ego_y = env.ego_vehicle.get_location().y
        ego_x = env.ego_vehicle.get_location().x
        ego_yaw = env.ego_vehicle.get_transform().rotation.yaw
        threshold = 10 if (205<ego_y<215) or (230<ego_y<240) else 1
        colli_risk = (abs(ego_x - 
                            waypoint(np.clip(ego_y,200,250)))>threshold)
        left_risk = (ego_x >338.5) and (ego_yaw<90)
        right_risk = (ego_x <335) and (ego_yaw>90)
            
        if not out_guide:
            x_i = 0
        if (colli_risk or left_risk or right_risk) and step!=0 and i%3==rand:
            xreal = scope['position_x']
            xref = waypoint(np.clip(scope['position_y'],200,250))
            x_i += (xreal - xref)
            action = np.clip (0.3*(xreal-xref)+ 0.0*(x_i), -1, 1)
            out_guide=True
        
        State_, action_fdbk, reward_e, _, done, scope = env.run_step(action, out_guide=out_guide)
        flag1.append(action_fdbk)
        
        if (action_fdbk is not None) or (out_guide is True):
            if flag_intervene == 0:
                reward_i = -10
                # reward_i = reward_i*0 if NUM>5 else reward_i
                flag_intervene += 1
            else:
                reward_i = 0
        else:
            reward_i = 0
            flag_intervene = 0
        

        reward = reward_e + reward_i
        
        reward_e_record[i].append(reward_e)
        reward_i_record[i].append(reward_i)
        reward_record[i].append(reward)
        
        x_per_episode[i].append(scope['position_x'])
        y_per_episode[i].append(scope['position_y'])
        
        # print(env.beyond)
        if step%control_freq==0:
            if action_fdbk is not None:
                if flag_qrecord==0 and (flag1[-2] is None):
                    bs = torch.tensor(State,dtype=torch.float).view(1,45,80).cuda()
                    ba = torch.tensor(DRL.actor(bs),dtype=torch.float).cuda()
                    q1,q2 = DRL.critic([bs,ba])
                    qlist.append([q1.detach().cpu().numpy(),q2.detach().cpu().numpy()])
                    
                action_disturbing_degree.append([action_fdbk,float(action)])
                intervene_percent_per_episode[i].append(action_fdbk)
                action = action_fdbk
                intervention = 1
                while flag<1:
                    DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
                    flag += 1
            elif out_guide is True:
                intervention = 1
                if flag_qrecord==0:
                    bs = torch.tensor(State,dtype=torch.float).view(1,45,80)
                    ba = torch.tensor(action,dtype=torch.float).unsqueeze(0).unsqueeze(0)
                    q1,q2 = DRL.critic([bs.cuda(),ba.cuda()])
                    qlist.append([q1.detach().cpu().numpy(),q2.detach().cpu().numpy()])
                    flag_qrecord=1
                DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
                intervene_percent_per_episode[i].append(action)
            else:
                intervention = 0
                DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
        
        State = State_

        final_action[i].append(action)
        
        if total_step > 256:
            c_loss, a_loss = DRL.learn(epoch=i)
            var *= 0.99988 if var>0.1 else 0.1
            # loss_critic.append(np.average(c_loss))
            loss_actor.append(np.average(a_loss))
            
        ep_reward += reward
        total_reward_list.append([reward_e,reward_i])

        total_step += 1
        step += 1
        
        if done:
            
            mean_reward =  ep_reward / step  
            mean_reward_list.append(ep_reward/step)
            real_reward_list.append(np.mean(reward_e_record[i]))
            duration_list.append(step)
            
            intervene_percent.append(len(intervene_percent_per_episode[i]))

            print('\n episode is:',i)
            # print('explore_rate:',round(var,4))
            print('c_loss:',round(np.average(c_loss),4))
            print('a_loss',round(np.average(a_loss),4))
            print('total_step:',total_step)
            print('episode_step:',step)
            
            writer.add_scalar('reward/reward_episode', mean_reward, i)
            writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), i)
            writer.add_scalar('reward/duration_episode', step, i)
            writer.add_scalar('percent_intervene', len(intervene_percent_per_episode[i]), i)
            writer.add_scalar('rate_exploration', round(var,4), i)
            writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), i)
            writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), i)

            # if i % 1 == 0:
            #     plt.ion()
            #     plt.subplot(311)
            #     plt.plot(reward_i_record[i])
            #     plt.plot(reward_record[i])
            #     plt.subplot(312)
            #     plt.plot(real_reward_list)
            #     plt.subplot(313)
            #     plt.plot(previous_action[i])
            #     plt.plot(final_action[i])
            #     plt.show()
            #     plt.pause(2)
            #     plt.close('all')
            
            break
        
        signal.signal(signal.SIGINT, signal_handler)


            
    if total_step>50000:
        break 

print('total time:',time.perf_counter()-start_time)        
        
DRL.save_model('./TD3_based_DRL/models')


pygame.display.quit()
pygame.quit()

a1 = previous_action[0:i]
a2 = final_action[0:i]
real_reward_list = np.squeeze(real_reward_list)
real_reward_list = real_reward_list.astype(np.float)
scio.savemat('dataHUG2{}.mat'.format(NUM), mdict={'a1': a1,'a2': a2,'step':duration_list,'reward':reward_record,'qlist':qlist,
                                                'intervenepercent':intervene_percent,'actiondisturbingdegree':action_disturbing_degree,
                                                'x':x_per_episode,'y':y_per_episode,'stepreward':total_reward_list,
                                                'r_i':reward_i_record,'r_e':reward_e_record})


