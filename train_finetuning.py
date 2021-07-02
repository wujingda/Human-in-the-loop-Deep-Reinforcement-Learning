import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import torch
import pygame

from env import scenario


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
    

seed = 2
np.random.seed(seed)


control_freq = 1
MAX_EPISODES = 30

total_step = 0

var = 0.5
tem = 0.5

flag = 0

a_loss,c_loss = 0,0

s_dim = 80*45
a_dim = 1

DRL = DRL(a_dim, s_dim)
DRL.critic.load_state_dict(torch.load('./TD3_based_DRL/models/critic.pkl'))
DRL.critic_target.load_state_dict(torch.load('./TD3_based_DRL/models/critic.pkl'))
### if pre-train
DRL.actor.load_state_dict(torch.load('./TD3_based_DRL/models/actor.pkl'))
DRL.actor_target.load_state_dict(torch.load('./TD3_based_DRL/models/actor.pkl'))


action = np.zeros(a_dim)
mean_reward_list = []
loss_critic = []
loss_actor = []
step_record =[]
loss = 0


previous_action = [[] for i in range(MAX_EPISODES)]
final_action = [[] for i in range(MAX_EPISODES)]

x_per_episode = [[] for i in range(MAX_EPISODES)]
y_per_episode = [[] for i in range(MAX_EPISODES)]

action_disturbing_degree = []
intervene_percent_per_episode = [[] for i in range(502)]
intervene_percent = []

data = [[] for i in range(MAX_EPISODES)]
total_reward_list = []

for i in range(MAX_EPISODES):

    mean_reward = 0
    ep_reward = 0
    step = 0
    v_tar = 10
    done = 0

    intervene_list = []

    judge_list = []

    r_record = []
    record1 = []
    
    flag = 0
    
    env = scenario(trainable=False)

    State, scope = env.obtain_observation()

    while True:
        
        if step % control_freq == 0:
            action = DRL.choose_action(State)
        else:
            action = final_action[i][-1] if total_step > 1 else 0.
        
        # if step%2==1:
        #     action = previous_action[i][-1]
        previous_action[i].append(action)
        
        State_, action_fdbk, reward_e, _, done, scope = env.run_step(action, out_guide=False)
        reward = reward_e

        if step % control_freq == 0:
            if action_fdbk is not None:
                action = action_fdbk
                # action = np.squeeze( np.clip( np.random.normal(action_fdbk,var), 0,1) )
                intervention = 1
                while flag<=1:
                    DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
                    flag += 1
            else:
                intervention = 0
                # DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)

        State = State_

        
        final_action[i].append(action)
        
        if total_step>1280:
            c_loss, a_loss = DRL.learn()
            loss_critic.append(np.average(c_loss))
            loss_actor.append(np.average(a_loss))
            
        ep_reward += reward
        r_record.append(reward)
        total_reward_list.append(reward.squeeze())
        
        x_per_episode[i].append(scope['position_x'])
        y_per_episode[i].append(scope['position_y'])
        
        total_step += 1
        step += 1
        
        if done:
            
            mean_reward =  ep_reward / step  
            mean_reward_list.append(ep_reward/step)
            step_record.append(step)
            
            intervene_percent.append(len(intervene_percent_per_episode[i]))

            print('\n episode is:',i)
            print('mean_reward:',np.round(mean_reward,4))
            print('explore_rate:',round(var,4))
            print('c_loss:',round(np.average(c_loss),4))
            print('steps:',step)
            print('a_loss',round(np.average(a_loss),4))
            print('total_step:',total_step)
            plt.subplot(311)
            plt.plot(r_record)
            plt.subplot(312)
            plt.plot(mean_reward_list)
            plt.subplot(313)
            plt.plot(previous_action[i])
            plt.plot(final_action[i])
            plt.show()
            
            if i % 1 == 0:
                torch.save(DRL.critic.state_dict(), './transitfinetune/criticfinetuningepoch{}.pth'.format(i)) 
                torch.save(DRL.actor.state_dict(), './transitfinetune/actorfinetuningepoch{}.pth'.format(i)) 
            break

    if total_step>75000 or i>30:
        break 
 
pygame.display.quit()
pygame.quit()

mean_reward_list = np.squeeze(mean_reward_list)
mean_reward_list = mean_reward_list.astype(np.float)
scio.savemat('finetuning.mat', mdict={'step':step_record,'reward':mean_reward_list,'stepreward':total_reward_list,
                                                'x':x_per_episode,'y':y_per_episode,'intervenepercent':intervene_percent,'actiondisturbingdegree':action_disturbing_degree,
                                                })


