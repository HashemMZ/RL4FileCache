#!/usr/bin/env python
# coding: utf-8

'''Author: HM
- training just one user that learns and other users exploit the strategy in the paper based on Puw probabilites
- reseting the env when done=1
- !major change: env.reset() is used instead of generating requests
- change user training loop to run a few times for each TX_file
- major v5-4: all users are behaved equally and find_decoding_candidates() is changed in env file
- v5-5: make changes in training loop in how the episodes and steps loops are structured
-v6: all users all trained not just user=0
- v7: just user0 is trained and other users are using the strategy in paper
- v9 make some changes to compare the results of other users and TRAINING_USER
- v10: changes in exp-replay buffer to fill with just success experiences
- !major v20: some important bugs of the code like the unwanted vanishing beta and cache contents are resolved.
- v21 counting satisfied requests, self requests and ... for comparison with paper
- v22: moved plot tasks into a funtion
- v24: adding LSTM layer
- removed lstm. used n step dqn method
- v25 nstep q learing
- v27 change structure based on cocolico14
- v28 alter filling buffer section to consider user0 as different user and save the connectivity of its state
- v29 changing the main sim loop with new structure
- using actor v3
- v29_2: change input to N items consisting beta*cache
'''

#%%Imports

import numpy as np
from numpy import random
import torch
import torch.nn as nn

from matplotlib import use
import matplotlib.pyplot as plt
from scipy.signal import convolve 
from scipy.signal.windows import gaussian
from envs.cache_env_v21_4 import CacheEnv, NULL
from actor.actor_v3_2 import actor, ReplayBuffer, DQNAgent



from tqdm import trange
from IPython.display import clear_output
##
#%% seed


# set a seed
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
mode = 'normal' #'debug' 
# use('agg')
#%% Torch check

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device=='cuda':
    print(torch.cuda.get_device_name())
   

# In[16]: ENVIRONMENT is defined here. all the parameters that are important is sent to the class init e.g. cache size, number of users and ...
################################################# 
#                    Env                        #
#################################################
#setup env and agent and target networks
zipf_param = [0.001, 0.25, 0.5, 1, 4];
zipf_index = 0
env_name = 'cacheEnv'
env = CacheEnv(cache_size = 50, number_of_users = 30, number_of_files_in_server = 100\
                   , Prob_req=0.2, req_pdf_param =zipf_param[zipf_index],\
                       replaceReward = -1, turnPenalty =-1 , winReward= 10)
state_dim = env.state_dim
n_actions = env.action_dim
N = env.N
K = env.K
M = env.M
state = env.reset()
# the distribution for generating requests. consider that the param is an attrib of env 
gamma = env.generate_distribution_samples(env.req_pdf_param)
# DQN has two networks a base network here called agent and a target network
agent = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
target_network = DQNAgent(state_dim, n_actions, epsilon=1).to(device)
# load the weights of base net into the target net
target_network.load_state_dict(agent.state_dict())
actor = actor(env, gamma = 0.999, n_step = 3)
TRAINING_USER = 0 #also called USER0 is the user that uses RL
#%% Filling replay buffer: in this section the buffer is filled with samples of state, action.... to start the training algorithm later. 

exp_replay = ReplayBuffer(10**7)
print('filling exprience-replay buffer with samples...\n')

env.reset()
t = 0
env.Prob_req = 0.2
while t in range(10 ** 6):
    # generate some requests
    [env.requests, ind_new_requests] = env.generate_requests(env.requests, gamma, env.Prob_req);
    
    for user in range(0,env.K):
        if env.requests[user]!= NULL() and env.caches[env.requests[user],user]==1:
            env.requests[user] = NULL();    #   Request is satisfied
            
    if any(env.requests!= NULL()):        
        # server prepare and send XOR file
        TX_file_set, count_satisfied_clique, count_satisfied_plain = actor.server_send()            
       
        for TX_file in TX_file_set:
                            
            for user in range(1,env.K):
                env.user_set(user) 
                if env.can_user_decode_file(TX_file, TX_file_set) and not env.caches[TX_file,user]:
                    env.update_cache(user, TX_file,  gamma)
        env.update_beta()    
        # training user or user0 
        user = TRAINING_USER
        env.user_set(user)
        for TX_file in TX_file_set:
            if env.can_user_decode_file(TX_file, TX_file_set) and not env.caches[TX_file,user]:
                actor.set_user_state(TX_file, TX_file_set, user)
                state, action, reward, next_state, done = actor.play(agent, env)
                actor.store(state, action, reward, next_state, done,  exp_replay )
                
                t +=1
                
                break
                
        env.requests[env.users_able_decoding] = NULL();  
        if len(exp_replay) >= 1* 10**2:
            break
    else:
        env.reset()

    
        
print(f'exp replay buffer is filled with {len(exp_replay)}')


#%% training param are set here
#######################################
#            training params
######################################
#setup some parameters for training
timesteps_per_epoch = 1
batch_size = 32
n_episodes =  5
total_steps = 1 * 10**4


#%%
#init Optimizer
opt = torch.optim.Adam(agent.parameters(), lr=1e-5)
# opt = torch.optim.AdamW(agent.parameters(), lr=1e-6, amsgrad=True)

# set exploration epsilon 
start_epsilon = 1
end_epsilon = 0.05
eps_decay_final_step = 2 * 10**4

# setup spme frequency for loggind and updating target network
loss_freq = 100
refresh_target_network_freq = 100
eval_freq = 1000
plot_freq = 50

# to clip the gradients
max_grad_norm = 100

mean_rw_history,td_loss_history =[],[]
loss = torch.as_tensor([0],device='cuda', dtype=torch.float32)


#%% some functions


def epsilon_schedule(start_eps, end_eps, step, final_step):
    return start_eps + (end_eps-start_eps)*min(step, final_step)/final_step

#%% related to plotthing results 
def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')
def plot_results(mean_rw_history,td_loss_history):

    
    
    fig = plt.figure(figsize=[16, 10])
    plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.subplot(2, 2, 3)
    plt.title("beta distribution of all users")
    plt.bar([i for i in range(N)],env.current_state['beta'])  
    plt.plot(gamma,color='red')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title("users' cache")
    #plt.bar([i for i in range(N)],env.current_state['cache'])
    plt.imshow(env.caches.transpose(),cmap='gray')
    # plt.grid()

    plt.subplot(2, 2, 1)
    plt.title("ratio of satisfied requests")
    plt.plot(mean_rw_history)
    plt.grid()

    assert not np.isnan(td_loss_history[-1])
    plt.subplot(2, 2, 2)
    plt.title("TD loss history (smoothened)")
    plt.plot(td_loss_history)
    plt.grid()

    plt.show()
    # plt.close(fig)


def train_user():
    # train by sampling batch_size of data from experience replay
    states, actions, rewards, next_states, done_flags = exp_replay.sample(batch_size)  

    # loss = <compute TD loss>
    loss = actor.compute_td_loss(agent, target_network, 
                           states, actions, rewards, next_states, done_flags,                                          
                           device=device)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()  
    return loss, grad_norm
#%% Main settings
# We now carryout the training on DQN setup above.

state = env.reset()
state_dim = env.state_dim
n_actions = env.action_dim
N = env.N
K = env.K
M = env.M
state = env.reset()



num_USER0_requests = 0
count_satisfied_clique_user0 = 0
count_satisfied_plain_user0 = 0
count_satisfied_self_user0 = 0
count_total_user0 = 0
    
count_satisfied_clique = 0
count_satisfied_plain = 0
count_satisfied_self = 0
count_total = 0
#%% Main loop
num_USER0_satisfied_requests = 0


reward = 0
sum_rewards = 0 
n = 20 # number of steps look forward
print('Training Loop started!\n')   
for episode in range(n_episodes):
    
  
    # resetting env fills caches with random files and generates some requests as well
    env.reset()
           
    sum_rewards  = 0  
    # num_USER0_requests = 0
    # num_USER0_satisfied_requests = 0
    t = 0
    tau = 0
    rewards, states, actions, episode_rewards = [],[],[],[]
    for t in  trange(total_steps + 1):
            
        # generate some requests
        [env.requests, ind_new_requests] = env.generate_requests(env.requests, gamma, env.Prob_req);
        
        #check requests already satisfied (in cache)
        for user in range(0,env.K):
            if env.requests[user]!= NULL() and env.caches[env.requests[user],user]==1:
                env.requests[user] = NULL();    #   Request is satisfied
                if user==TRAINING_USER:
                    count_satisfied_self_user0 += 1
                else:
                    count_satisfied_self += 1
                        
                
        if any(env.requests!= NULL()):
    
            # server prepare and send XOR file
            TX_file_set, num_satisfied_clique, num_satisfied_plain = actor.server_send()            
            if num_satisfied_clique:
                if TRAINING_USER in env.users_able_decoding: 
                    count_satisfied_clique_user0 += 1
                    count_satisfied_clique += num_satisfied_clique-1 #removing user0
                else:
                    count_satisfied_clique += num_satisfied_clique
            elif num_satisfied_plain:
                if TRAINING_USER in env.users_able_decoding: 
                    count_satisfied_plain_user0 += 1
                    count_satisfied_plain += num_satisfied_plain-1 #removing user0
                else:
                    count_satisfied_plain += num_satisfied_plain
                    
            # reduce exploration as we progress
            agent.epsilon = epsilon_schedule(start_epsilon, end_epsilon, episode, eps_decay_final_step)
               
            #-----------------------------------------
          
            # loop for other users
            for TX_file in TX_file_set:
                                
                for user in range(1,env.K):
                    env.user_set(user) 
                    if env.can_user_decode_file(TX_file, TX_file_set) and not env.caches[TX_file,user]:
                        env.update_cache(user, TX_file,  gamma)
            env.update_beta()    
            # training user or user0 
            user = TRAINING_USER
            env.user_set(user)
            for TX_file in TX_file_set:
                if env.can_user_decode_file(TX_file, TX_file_set) and not env.caches[TX_file,user]:
                    actor.set_user_state(TX_file, TX_file_set, user)
                    state, action, reward, next_state, done = actor.play(agent, env)
                    actor.store(state, action, reward, next_state, done,  exp_replay )
                    
                    if done:
                        # T = t+1
                        sum_rewards += reward  
                        num_USER0_satisfied_requests += 1#
                        loss,grad_norm = train_user()
                        
                        # print(f'\nloss: {loss}\n')
                    
                    if t % loss_freq == 0:
                        td_loss_history.append(loss.data.cpu().item())
                        mean_rw_history.append(num_USER0_satisfied_requests/(count_total_user0+1))
                        #save grad_norm for plotting
                
                    if t % refresh_target_network_freq == 0:
                        # Load agent weights into target_network
                        target_network.load_state_dict(agent.state_dict())
                    if t % plot_freq == 0:
                        clear_output(True)
                        # plot_results(mean_rw_history,td_loss_history)
                    #requests for these users is satisfied
                    
                    break
            env.requests[env.users_able_decoding] = NULL();       
             
        else:
            env.reset()
            
        if TRAINING_USER in ind_new_requests:
            count_total_user0 += 1
            count_total += len(ind_new_requests)-1    
        else:
            count_total += len(ind_new_requests)
    plot_results(mean_rw_history,td_loss_history)
#%% results
print(f'\nzipf param = {zipf_param[zipf_index]}')
print(f'count_satisfied_clique_user0*(K-1) = {count_satisfied_clique_user0*(K-1)}\
          \ncount_satisfied_plain_user0*(K-1) = {count_satisfied_plain_user0*(K-1)}\
              \ncount_satisfied_self_user0*(K-1) = {count_satisfied_self_user0*(K-1)}\
              \n\ncount_satisfied_clique = {count_satisfied_clique}\
                  \ncount_satisfied_plain = {count_satisfied_plain}\
                      \ncount_satisfied_self = {count_satisfied_self}\n\n')
print(f'total requests = {count_total}\ntotal user0 requests*(K-1) = {count_total_user0*(K-1)}') 
torch.save(agent.state_dict(), f'.\\agent{zipf_param[zipf_index]}')
plot_results(mean_rw_history,td_loss_history)