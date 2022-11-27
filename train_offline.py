import numpy as np
import time
import scipy.io as scio
import os
import pygame
import signal
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('./TD3_based_DRL/checkpoints/log')

# import CARLA environment
from env import scenario

# import associated tools
from utils import set_seed, signal_handler, get_path, RND


def RL_training():
    
    set_seed(args.seed)
    
    if args.algorithm == 0:
        from TD3_based_DRL.TD3HUG import DRL
        log_dir = 'TD3_based_DRL/checkpoints/TD3HUG.pth'
    elif args.algorithm == 1:
        from TD3_based_DRL.TD3IARL import DRL
        log_dir = 'TD3_based_DRL/checkpoints/TD3IARL.pth'
    elif args.algorithm == 2:
        from TD3_based_DRL.TD3HIRL import DRL   
        log_dir = 'TD3_based_DRL/checkpoints/TD3HIRL.pth'
    else:
        from TD3_based_DRL.TD3 import DRL    
        log_dir = 'TD3_based_DRL/checkpoints/TD3.pth'
    
    env = scenario()
    
    s_dim = [env.observation_size_width, env.observation_size_height]
    a_dim = env.action_size
    DRL = DRL(a_dim, s_dim)
    
    if args.reward_shaping == 3:
        rnd = RND()
    
    if args.resume and os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        DRL.load(log_dir)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    exploration_rate = args.initial_exploration_rate
    
    
    # initialize measurable variables
    total_step = 0
    a_loss,c_loss = 0,0
    
    loss_critic, loss_actor = [], []
    
    episode_reward_list, global_reward_list, episode_duration_list = [], [], []
    
    previous_action = [[] for i in range(args.maximum_episode)]
    final_action = [[] for i in range(args.maximum_episode)]
    
    # reward_i: virtual reward (added shaping term); reward_e: real reward
    reward_i_record = [[] for i in range(args.maximum_episode)]
    reward_e_record = [[] for i in range(args.maximum_episode)]
    
    # calculate human intervention rate
    action_disturbing_degree = []
    intervene_percent_per_episode = [[] for i in range(args.maximum_episode)]
    intervene_percent = []
    
    # record the x,y coordinates of the ego vehicle
    x_per_episode = [[] for i in range(args.maximum_episode)]
    y_per_episode = [[] for i in range(args.maximum_episode)]
    
    # record the q value difference
    qlist = []
    
    path_generator = get_path()
    
    env = scenario()
    
    start_time = time.perf_counter()
    
    for i in range(start_epoch, args.maximum_episode):
        reward = 0
        ep_reward = 0      
        step = 0
        step_intervene = 0
    
        done = False
    
        list_fdbk = []
        list_fdbk.append(None)
        
        flag_qrecord = 0
    
        pid_activation = 0
        pid_seed = np.random.randint(0,3)
        pid_intergal_value = 0
        
        State, scope = env.restart()
    
        while True:
            ## Section DRL's actting ##
            action = DRL.choose_action(State)
            # add a Gaussian noise to the DRL action
            action = np.clip( np.random.normal(action, exploration_rate), -1, 1)
    
            previous_action[i].append(action)
            ## End of Section DRL's actting
        
        
            ## Section PI controller (can sometimes substitute real human participants) ##
            if args.pid_controller_guidance:
                
                # calculate some indicators of the environment to determine the activation of the PI controller
                ego_y = env.ego_vehicle.get_location().y
                ego_x = env.ego_vehicle.get_location().x
                ego_yaw = env.ego_vehicle.get_transform().rotation.yaw
                threshold = 10 if (205 < ego_y < 215) or (230 < ego_y < 240) else 1
                colli_risk = (abs(ego_x - path_generator(np.clip(ego_y, 200, 250))) > threshold)
                left_risk = (ego_x > 338.5) and (ego_yaw < 90)
                right_risk = (ego_x < 335) and (ego_yaw > 90)
                
                if not pid_activation:
                    pid_intergal_value = 0
                
                if (colli_risk or left_risk or right_risk) and (step != 0) and (i % 3 == pid_seed):
                    pid_activation = True
                    xreal = scope['position_x']
                    xref = path_generator(np.clip(scope['position_y'], 200, 250))
                    pid_intergal_value += (xreal - xref)
                    pid_proportional = xreal - xref
                    action = np.clip(0.3 * (pid_proportional) + 0.0 * (pid_intergal_value), -1, 1)
                else:
                    pid_activation = False
            ## End of Section pid controller ##
    
    
            ## Section environment update ##
            State_, action_fdbk, reward_e, _, done, scope = env.run_step(action)
            # action_fdbk is not None if human participants manipulate the steering wheel
            list_fdbk.append(action_fdbk)
            ## End of Section environment update ##
    
    
            ## Section reward shaping ##
            # intervention penalty-based shaping
            if args.reward_shaping == 1:
                # only the 1st intervened time step is penalized
                if (action_fdbk is not None) or (pid_activation is True):
                    if step_intervene == 0:
                        reward_i = -10
                        step_intervene += 1
                    else:
                        reward_i = 0
                else:
                    reward_i = 0
                    step_intervene = 0
            # heuristic potential-based shaping
            elif args.reward_shaping == 2:
                reward_i = 250 - ego_y
            # RND shaping
            elif args.reward_shaping == 3:
                error, mu, std = rnd.forward(State_)
                reward_i = (min(max(1 + (error - mu) / std, 0.35), 2) - 0.35) * 10
            # no shaping
            else:
                reward_i = 0
            reward = reward_e + reward_i
            ## End of Section reward shaping ##
    
    
            ## Section DRL store ##
            # human intervention event occurs
            if action_fdbk is not None:
                # record the q difference of the 1st time step of one human intervention event
                if (flag_qrecord == 0) and (list_fdbk[-2] is None):
                    bs = torch.tensor(State,dtype=torch.float).view(1, env.observation_size_height, env.observation_size_width).to(args.device)
                    ba = torch.tensor(DRL.actor(bs),dtype=torch.float).to(args.device)
                    q1, q2 = DRL.critic([bs, ba])
                    qlist.append([q1.detach().cpu().numpy(), q2.detach().cpu().numpy()])
                # record the action difference of one human intervention event    
                action_disturbing_degree.append([action_fdbk, float(action)])
                intervene_percent_per_episode[i].append(action_fdbk)
    
                action = action_fdbk
    
                intervention = 1
                DRL.store_transition(State, action, action_fdbk, intervention, reward, State_)
            # PI controller event occurs
            elif pid_activation is True:
                intervention = 1
                if flag_qrecord == 0:
                    bs = torch.tensor(State,dtype=torch.float).view(1, env.observation_size_height, env.observation_size_width).to(args.device)
                    ba = torch.tensor(action,dtype=torch.float).unsqueeze(0).unsqueeze(0).to(args.device)
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
            learn_threshold = args.warmup_threshold if args.warmup else 256
            if total_step > learn_threshold:
                c_loss, a_loss = DRL.learn(epoch=i)
                loss_critic.append(np.average(c_loss))
                loss_actor.append(np.average(a_loss))
                # Decrease the exploration rate
                exploration_rate = exploration_rate * args.exploration_decay_rate if exploration_rate>args.cutoff_exploration_rate else 0.05
            ## End of Section DRL update ##
    
    
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
                # print('explore_rate:',round(exploration_rate,4))
                # print('c_loss:',round(np.average(c_loss),4))
                # print('a_loss',round(np.average(a_loss),4))
                # print('total_step:',total_step)
                # print('episode_step:',step)
                
                writer.add_scalar('reward/reward_episode', mean_reward, i)
                writer.add_scalar('reward/reward_episode_noshaping', np.mean(reward_e_record[i]), i)
                writer.add_scalar('reward/duration_episode', step, i)
                writer.add_scalar('percent_intervene', len(intervene_percent_per_episode[i]), i)
                writer.add_scalar('exploration_rate', round(exploration_rate,4), i)
                writer.add_scalar('loss/loss_critic', round(np.average(c_loss),4), i)
                writer.add_scalar('loss/loss_actor', round(np.average(a_loss),4), i)
                
                break
            
            signal.signal(signal.SIGINT, signal_handler)
    
                
        if total_step > args.maximum_step:
            break 
    
    print('total time:',time.perf_counter()-start_time)        
            
    DRL.save_model('./TD3_based_DRL/models')
    
    pygame.display.quit()
    pygame.quit()
    
    action_drl = previous_action[0:i]
    action_final = final_action[0:i]
    scio.savemat('data{}-{}.mat'.format(args.algorithm,round(time.time())), mdict={'action_drl': action_drl,'action_final': action_final,
                                                    'actiondisturbingdegree':action_disturbing_degree,
                                                    'qlist':qlist,'intervenepercent':intervene_percent,
                                                    'x':x_per_episode,'y':y_per_episode,'stepreward':global_reward_list,
                                                    'step':episode_duration_list,'reward':episode_reward_list,
                                                    'r_i':reward_i_record,'r_e':reward_e_record})
    
    





if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--algorithm', type=int, help='RL algorithm (0 for Proposed, 1 for IARL, 2 for HIRL, 3 for Vanilla TD3) (default: 0)', default=0)
    parser.add_argument('--maximum_episode', type=float, help='maximum training episode number (default:1000)', default=1000)
    parser.add_argument('--maximum_step', type=float, help='maximum training step number (default:5e4)', default=5e4)
    parser.add_argument('--seed', type=int, help='fix random seed', default=2)
    parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
    parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
    parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
    parser.add_argument('--resume', action="store_true", help='whether to resume trained agents (default: False)', default=False)
    parser.add_argument('--warmup', action="store_true", help='whether to start training until collecting enough data (default: False)', default=False)
    parser.add_argument('--warmup_threshold', type=int, help='warmup length by step (default: 5000)', default=5e3)
    parser.add_argument('--pid_controller_guidance', action="store_true", help='whether to use PID controller providing guidance action (default: False)', default=False)
    parser.add_argument('--reward_shaping', type=int, help='reward shaping scheme (0: none; 1:intervention-based; 2:potential-based; 3: RND-based) (default: 0)', default=0)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    # Run
    RL_training()
