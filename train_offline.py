import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os
import pygame
import signal

import torch
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./TD3_based_DRL/checkpoints/log')

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### Import CARLA environment
from env import scenario

from utils import signal_handler, get_path, RND
rnd = RND()

### Arguments
MAX_EPISODES = 1500 
MAX_STEPS = 5e4
LOAD = 0
PID = 0
REWARD_SHAPING = 1
PRE_EXPLORATION = 1
PRE_INIT_CRITIC = 0

### Construct the DRL agent
device = torch.device('cuda') if torch.cuda.is_available() else ('cpu')
algorithm = 0

from TD3_based_DRL.Priority_Replay import Memory
if algorithm == 0:
    from TD3_based_DRL.TD3HUG import DRL
    log_dir = 'TD3_based_DRL/checkpoints/TD3jingda.pth'
elif algorithm == 1:
    from TD3_based_DRL.TD3IARL import DRL
    log_dir = 'TD3_based_DRL/checkpoints/TD3IARL.pth'
elif algorithm == 2:
    from TD3_based_DRL.TD3HIRL import DRL   
    log_dir = 'TD3_based_DRL/checkpoints/TD3HIRL.pth'
else:
    from TD3_based_DRL.TD3 import DRL    
    log_dir = 'TD3_based_DRL/checkpoints/TD3.pth'

s_dim = [env.observation_size_width, env.observation_size_height]
a_dim = env.action_size
DRL = DRL(a_dim, s_dim)

 if PRE_INIT_CRITIC:
     DRL.memory_load()
     for _ in range(10000):
         DRL.pre_init_critic()

if os.path.exists(log_dir) and LOAD:
    checkpoint = torch.load(log_dir)
    DRL.load(log_dir)
    start_epoch = checkpoint['epoch'] + 1

var = 0.5  # initial exploration rate


### Initialize measurable variables
total_step = 0
a_loss,c_loss = 0,0
start_epoch = 0

loss_critic, loss_actor = [], []

episode_reward_list, global_reward_list, episode_duration_list = [], [], []

previous_action = [[] for i in range(MAX_EPISODES)]
final_action = [[] for i in range(MAX_EPISODES)]

# reward_i: virtual reward (added shaping term); reward_e: real reward
reward_i_record = [[] for i in range(MAX_EPISODES)]
reward_e_record = [[] for i in range(MAX_EPISODES)]

# calculate human intervention rate
action_disturbing_degree = []
intervene_percent_per_episode = [[] for i in range(MAX_EPISODES)]
intervene_percent = []

# record the x,y coordinates of the ego vehicle
x_per_episode = [[] for i in range(MAX_EPISODES)]
y_per_episode = [[] for i in range(MAX_EPISODES)]

# record the q value difference
qlist = []

path_generator = get_path()

env = scenario(random_spawn = False)

start_time = time.perf_counter()

for i in range(start_epoch, MAX_EPISODES):

    ep_reward = 0      
    step = 0
    step_intervene = 0

    done = False

    list_fdbk = []
    list_fdbk.append(None)
    
    flag_qrecord = 0

    pid_guide = False
    pid_seed = np.random.randint(0,3)
    pid_intergal = 0

    if i == start_epoch:
        env.destroy()

    env = scenario(random_spawn = False)
    State, scope = env.obtain_observation()

    while True:
        ## Section DRL's actting ##
        action = DRL.choose_action(State)
        # add a Gaussian noise to the DRL action
        action = np.clip( np.random.normal(action, var), -1, 1)

        previous_action[i].append(action)
        ## End of Section DRL's actting
    

        ## Section PI controller (can sometimes substitute real human participants) ##
        # calculate some indicators of the environment to determine the activation of the PI controller
        ego_y = env.ego_vehicle.get_location().y
        ego_x = env.ego_vehicle.get_location().x
        ego_yaw = env.ego_vehicle.get_transform().rotation.yaw
        threshold = 10 if (205 < ego_y < 215) or (230 < ego_y < 240) else 1
        colli_risk = (abs(ego_x - path_generator(np.clip(ego_y, 200, 250))) > threshold)
        left_risk = (ego_x > 338.5) and (ego_yaw < 90)
        right_risk = (ego_x < 335) and (ego_yaw > 90)
        
        if not pid_guide:
            pid_intergal = 0
        
        if (colli_risk or left_risk or right_risk) and (step != 0) and (i % 3 == pid_seed):
            pid_guide = True
            xreal = scope['position_x']
            xref = path_generator(np.clip(scope['position_y'], 200, 250))
            pid_intergal += (xreal - xref)
            pid_proportional = xreal - xref
            action = np.clip(0.3 * (pid_proportional) + 0.0 * (pid_intergal), -1, 1)
        else:
            pid_guide = False
        ## End of Section pid controller ##


        ## Section environment update ##
        State_, action_fdbk, reward_e, _, done, scope = env.run_step(action)
        # action_fdbk is not None if human participants manipulate the steering wheel
        list_fdbk.append(action_fdbk)
        ## End of Section environment update ##


        ## Section reward shaping ##
        # intervention penalty-based shaping
        elif REWARD_SHAPING == 1:
            # only the 1st intervened time step is penalized
            if (action_fdbk is not None) or (pid_guide is True):
                if step_intervene == 0:
                    reward_i = -10
                    step_intervene += 1
                else:
                    reward_i = 0
            else:
                reward_i = 0
                step_intervene = 0
        # heuristic potential-based shaping
        elif REWARD_SHAPING == 2:
            reward_i = 250 - ego_y
        # RND shaping
        elif REWARD_SHAPING == 3:
            error, mu, std = rnd.forward(State_)
            reward_i = (min(max(1 + (error - mu) / std, 0.35), 2) - 0.35) * 10
        # no shaping
        else:
            reward_i = 0
        ## End of Section reward shaping ##


        ## Section DRL store ##
        # human intervention event occurs
        if action_fdbk is not None:
            # record the q difference of the 1st time step of one human intervention event
            if (flag_qrecord == 0) and (list_fdbk[-2] is None):
                bs = torch.tensor(State,dtype=torch.float).view(1, env.observation_size_height, env.observation_size_width).to(device)
                ba = torch.tensor(DRL.actor(bs),dtype=torch.float).to(device)
                q1, q2 = DRL.critic([bs, ba])
                qlist.append([q1.detach().cpu().numpy(), q2.detach().cpu().numpy()])
            # record the action difference of one human intervention event    
            action_disturbing_degree.append([action_fdbk, float(action)])
            intervene_percent_per_episode[i].append(action_fdbk)

            action = action_fdbk

            intervention = 1
            DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
        # PI controller event occurs
        elif pid_guide is True:
            intervention = 1
            if flag_qrecord == 0:
                bs = torch.tensor(State,dtype=torch.float).view(1, env.observation_size_height, env.observation_size_width).to(device)
                ba = torch.tensor(action,dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
                q1, q2 = DRL.critic([bs, ba])
                qlist.append([q1.detach().cpu().numpy(), q2.detach().cpu().numpy()])
                flag_qrecord = 1
            
            intervene_percent_per_episode[i].append(action)

            DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
        # No intervention occurs
        else:
            intervention = 0
            DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
        ## End of DRL store ##


        ## Section DRL update ##
        learn_threshold = 5000 if PRE_EXPLORATION else 256
        if total_step > learn_threshold:
            c_loss, a_loss = DRL.learn(epoch=i)
            loss_critic.append(np.average(c_loss))
            loss_actor.append(np.average(a_loss))
            # Decrease the exploration rate
            var = var * 0.99988 if var>0.05 else 0.05
        ## End of Section DRL update ##


        reward = reward_e + reward_i
        ep_reward += reward
        global_reward_list.append([reward_e,reward_i])
        reward_e_record[i].append(reward_e)
        reward_i_record[i].append(reward_i)
        
        final_action[i].append(action)

        x_per_episode[i].append(scope['position_x'])
        y_per_episode[i].append(scope['position_y'])

        State = State_

        total_step += 1
        step += 1
        
        if done:
            
            mean_reward =  ep_reward / step  
            episode_reward_list.append(mean_reward)
            episode_duration_list.append(step)
            intervene_percent.append(len(intervene_percent_per_episode[i]))

            # print('\n episode is:',i)
            # print('explore_rate:',round(var,4))
            # print('c_loss:',round(np.average(c_loss),4))
            # print('a_loss',round(np.average(a_loss),4))
            # print('total_step:',total_step)
            # print('episode_step:',step)
            
            writer.add_scalar('reward/reward_episode', mean_reward, i)
            writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), i)
            writer.add_scalar('reward/duration_episode', step, i)
            writer.add_scalar('percent_intervene', len(intervene_percent_per_episode[i]), i)
            writer.add_scalar('rate_exploration', round(var,4), i)
            writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), i)
            writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), i)
            
            break
        
        signal.signal(signal.SIGINT, signal_handler)

            
    if total_step > MAX_STEPS:
        break 

print('total time:',time.perf_counter()-start_time)        
        
DRL.save_model('./TD3_based_DRL/models')

pygame.display.quit()
pygame.quit()

action_drl = previous_action[0:i]
action_final = final_action[0:i]
scio.savemat('data{}-{}.mat'.format(algorithm,round(time.time())), mdict={'action_drl': action_drl,'action_final': action_final,
                                                'actiondisturbingdegree':action_disturbing_degree,
                                                'qlist':qlist,'intervenepercent':intervene_percent,
                                                'x':x_per_episode,'y':y_per_episode,'stepreward':global_reward_list,
                                                'step':episode_duration_list,'reward':episode_reward_list,
                                                'r_i':reward_i_record,'r_e':reward_e_record})


