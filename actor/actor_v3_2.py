# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:10:38 2024

@author: HM
- v2: removed LSTM
- v3: changed td_loss to correct bug in Sanghi code ( as well as pytorch website) the process to calc next state and target network is completely changed
- forward method of nn is changed to user tensor calculations
- work with v29_2 inputs are N items beta*cache
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque

def NULL():
    return -1
#%% actor is a class to manage the relation between env and RL algorithm.
class actor:
    def __init__(self, env, gamma = 0.99, n_step = 3):
        self.env = env
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.alpha = 0.6 # Amount of Prioritization
        self.gamma = 0.99 # Discount Factor
        self.epsilon = 1.0 # Max Prob for Explore
        self.epsilon_min = 0.1 # Min Prob for Explore
        self.epsilon_decay = 0.995 # Decay Rate for Epsilon
        self.update_rate = 1000  # Freq of Network Update
    
    def format_input_for_nn(self):
        env = self.env
        
        s = env.current_state
        user = s['user']
        cache = env.caches[:,user]
        TX_file = s['TX_file']
        #input to the neural net is prepared by stacking cache of user and beta (distribution)
        input_s =  cache.copy()
        input_s[TX_file] = 1
        # if user has some request set its equivalent input in input_s to some positive number say 10
        request = env.requests[user]
        if request != NULL():
            input_s[request] = 1
        input_s =  np.multiply(input_s,s['beta'])
        return input_s
    
    def set_user_state(self,TX_file, TX_file_set, user):
        env = self.env
        env.current_state['TX_file'] = TX_file 
        env.set_TX_file_set(TX_file_set)
   
    def store(self,state, action, reward, next_state, done,  exp_replay ):
        # implementation of n-step
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if done or len(self.n_step_buffer) == self.n_step:
            for transition in reversed(self.n_step_buffer):
                _,_,r, n_s, d = transition
    
                reward = r + self.gamma * reward * (1 - d)
                if d == 1: # this state is the last state for the current state 
                    next_state, done, reward = (n_s, d,r) 
    
            # change the state and ... to the first instant in the n_step_buffer (state, action, reward, next_state, done)
            state, action = self.n_step_buffer[0][:2]
            #clear buffer for next request round        
            self.n_step_buffer.clear()
            
            exp_replay.add(state, action, reward, next_state, done)              
        
    
    def play(self,agent, env ):
        input_s = self.format_input_for_nn()
    
        qvalues = agent.get_qvalues(input_s)
        action = agent.sample_actions(qvalues)[0]        
        output_s, r, done = env.step(action)          
            
        return input_s,action, r,output_s,done
    
    def play_and_record(self,agent, env, exp_replay, n_steps = 1 ):
        input_s = self.format_input_for_nn()
        sum_rewards = 0
        for _ in range(n_steps):
            qvalues = agent.get_qvalues(input_s)
            action = agent.sample_actions(qvalues)[0]        
            output_s, r, done = env.step(action)          
            sum_rewards += r
            self.store(input_s, action, r, output_s, done, exp_replay)
            if done:
                env.reset()
            else:
                input_s = output_s
            
        return input_s,action, r,output_s,done
    
    #%% server side: prepares a set of files called TX_file_set and send it to users. whether clique size is larger or number of users that want a specific plain file
    # this file set is prepared.
    def server_send(self):
        env = self.env
        usr_clique = env.find_largest_clique(env.requests, env.caches) #Find largest clique
        clique_size = len(usr_clique) 
        [plain_file_max, plain_file_req_size] = env.find_most_wanted_plain_file(env.requests)
        
        count_satisfied_clique = 0
        count_satisfied_plain = 0
        if clique_size > plain_file_req_size :         # If the clique is larger than the highest plain requests...
            TX_file_set = env.requests[usr_clique]
            TX_file_set = np.unique(TX_file_set)
            users_able_decoding = usr_clique   #... transmit the coded message corresponding to largest clique and...             
            count_satisfied_clique =  len(TX_file_set)
            
        else: # ... transmit most popular file
            TX_file_set = (plain_file_max,) 
            users_able_decoding = np.where(env.requests == plain_file_max)[0]
            count_satisfied_plain = plain_file_req_size

        env.users_able_decoding = users_able_decoding   
        #print(f'users_able_decoding={users_able_decoding}')
        return TX_file_set, count_satisfied_clique, count_satisfied_plain
    #%% td_loss is calculated in each training step for optimizing NN weights, this is based on time-delay method in which network output in different times is compared
    # and the loss is calculated based on them

    def compute_td_loss(self, agent, target_network, states, actions, rewards, next_states, done_flags, device='cuda'):
        gamma = self.gamma
        # convert numpy array to torch tensors
        states = torch.tensor(states, device=device, dtype=torch.float)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float)
        done_flags = torch.tensor(done_flags.astype('float32'),device=device,dtype=torch.float)
        
        # get q-values for all actions in current states
        # use agent network
        predicted_qvalues = agent(states)
        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]#,_ = torch.max(predicted_qvalues, dim=1)# 
       
        # predicted next state action
        predicted_next_qvalues_model = agent(next_states)
        # select q-values for chosen actions
        _,predicted_next_actions_model_ind = torch.max(predicted_next_qvalues_model, dim=1)# [range(len(actions)), actions]#,_ = torch.max(predicted_qvalues, dim=1)# 
        
        # compute q-values for all actions in next states
        # use target network
        with torch.no_grad():
            predicted_next_qvalues = target_network(next_states)
        
        # compute Qmax(next_states, actions) using predicted next q-values
        # we have to use predicted_next_actions_model to find 
        #next_state_values,_ = torch.max(predicted_next_qvalues, dim=1)
        next_state_values = predicted_next_qvalues.gather(1,predicted_next_actions_model_ind.view(-1,1))
        next_state_values = torch.squeeze(next_state_values,1)
        # compute "target q-values" 
        target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)
        

        return loss
#%% DQN class


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0, device='cuda'):
        
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        state_dim = state_shape#[0]
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.layer1 = nn.Sequential(
            nn.Linear(state_dim, 128),#128
            # nn.LayerNorm(128),
            nn.ReLU())
        self.layer2 = nn.Sequential(    
            nn.Linear(128, 100),
            # nn.LayerNorm(100),
            nn.ReLU())
            
        self.layerRNN = nn.LSTM(100,100,5)
        self.layerOut = nn.Sequential(    
            nn.Linear(128, n_actions),
            # nn.LogSigmoid()          
            )
        # 
        #self.parameters = self.network.parameters
        # forward function calculates the forward path of nn.Module
    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.layer1(state_t)
        # qvalues = self.layer2(qvalues)
        # qvalues,_ = self.layerRNN(qvalues)
        qvalues = self.layerOut(qvalues)
        # indices that are already zero in input state={cache+decoded file+request} cannot be used; they are masked out
        qvalues = torch.mul(qvalues,state_t) # the only way to avoid backpropg error
        return qvalues 
    
    


    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues,batch_size=1):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        
        #print(f'mean of qvalues = {np.mean(qvalues)}')
        valid_indices = np.where(qvalues!=0)[0]
        random_actions = np.random.choice(valid_indices, size=batch_size)
        
        best_actions_in_valid_indices = qvalues[valid_indices].argmax(axis=-1)
        best_actions = valid_indices[best_actions_in_valid_indices]
        
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
#

    #%% replay buffer: this buffer is filled with some (state-action-reward-next state ) data sets that is later used for training network (taking out random batched from it)
    # this is necessary for DQN networks to avoid learning individual pathes in the trajectory


class ReplayBuffer:
    def __init__(self, size):
        self.size = size #max number of items in buffer
        self.buffer =[] #array to holde buffer
        self.next_id = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
           self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size
        
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)