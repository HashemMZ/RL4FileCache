"""Caching full length files Environment
Custom Environment for the caching problem

Runtime: Python 3.6.5
Dependencies: None
DocStrings: NumpyStyle

Author : HM
- change all the cache storing to binary coded ones to avoid mistakes
- adding gamma and Prov_req to env params
- chaging beta to be in [0,1] by dividing by MK. later just devided by M to increase values toward 1. 
- !major: v3 in step funtion and in beta and ...
- v4: added state diagram to next_state()
- v5: can_decode_file had a bug which removed using np.unique
- !major v6: changes in decison process tree in step()
- !major v21: totally changed the reward section based on some simple assumptions
- !major v21: corrections in calculation of metric. there was a bug beta[cache] corrected to beta*cache. due to conversion from matlab 
- v21: removed unnecessary inputs of env class
- v21_2: change can_decode_file to do not check if the file is in cache
- change step to evict tx_file if location add and remove are the same
- to work with v_29_2 input is N items
"""
import numpy as np 
import matplotlib.pyplot as plt
from random import shuffle


def NULL():
    '''defined for null requests'''
    return -1
class CacheEnv():
    """Grid-World Environment Class
    """
    def __init__(self, cache_size=5, number_of_users = 3, number_of_files_in_server=10, 
                 Prob_req=0.2, req_pdf_param =.04,\
                     replaceReward = 0, turnPenalty = -1, winReward= 10, wrongReward=-10):
        """Initialize an instance of Grid-World environment
        """

        self.K = number_of_users
        self.N = number_of_files_in_server
        self.M = cache_size

        self.winReward = winReward
        self.replaceReward = replaceReward
        self.turnPenalty = turnPenalty

        self.current_state = {}
        self.NULL_REQ = NULL()
        self.requests = np.ones(shape=self.K,dtype=int)*NULL()
        self.current_state['cache'] = np.zeros(cache_size)
        self.current_state['beta'] =[0]
        self.current_state['user'] = 0 #index for user under consideration
        #self.current_state['can_decode_flag'] = False #shows whether user can decode TX_file or not             
        self.current_state['TX_file']=0
        self.current_state['TX_file_set']  = {}
        
        self.caches = np.zeros([self.N,self.K],dtype=(int))
        self.state_dim = self.N
        self.action_dim= self.N#+1 # existing N files that from which M can be in the cache plus a 'do_nothing' action for when the file is in the cache
        self.req_pdf_param = req_pdf_param 
        self.Prob_req = Prob_req
        self.decoded_file_binary_index = 0
        self.decoded_file = 0
        self.users_able_decoding = []

        self.reward = {'success':winReward,
                       'some-decode':replaceReward,
                       'no-decode':turnPenalty,
                       'no-success':wrongReward}
                       
        self.do_nothing_action = self.N;


#%%
    def reset(self):
        """reset
        """
        self.caches = self.randperm(self.M,self.K, self.N,method='binary')
        #self.convert_1hot_to_numerical_caches()
        self.current_state['beta'] = self.calculate_beta()
        self.decoded_file = 0
      
        zipf = self.generate_distribution_samples(self.req_pdf_param)
        self.requests,ind = self.generate_requests(np.ones(shape=[self.K],dtype=(int))*NULL(),zipf,0)
        
        
        self.isGameEnd = [0]*self.K
        self.totalAccumulatedReward = 0
        self.totalTurns = 0  
        #self.current_state['cache'] = self.caches[:,self.current_state['user']]

        return self.current_state
    
#%%
    def step(self,action):
        """
        Returns (next_state, instantaneous_reward, done_flag, info)
        action is of size 2 and shows the indices to remove from cache
        it updates beta, caches"""                
        
        location_remove = action
        location_add = self.current_state['TX_file']
        user = self.current_state['user']
        cache = self.caches[:,user]
        
        request = self.requests[user]
        reward = 0  
        done = 0
        
        if cache[location_remove]==1:# and cache[location_add]==0: # a file in cache should be replaced
            #update cache and beta
            self.caches[location_remove,user]=0                
            self.caches[location_add,user]=1
            self.current_state['beta']=self.calculate_beta()
            reward += self.reward['some-decode']
        
        if request != NULL() and cache[request]:     
            reward += self.reward['success']
            self.requests[user] = NULL()                
            done = 1
        output_s =  np.multiply(self.caches[:,user],self.current_state['beta'])   
        return output_s, reward, done
    


        

 #%%   
    def prepare_formatted_state(self,user,file_decoded):
        '''Prepare formatted state for the input of NN'''
        
        cache_decFile_req_state = self.caches[:,user].copy()
        
        self.decoded_file_binary_index = file_decoded-1;# make the entry related to the decoded file equal to 1 in the state variable
        cache_decFile_req_state[self.decoded_file_binary_index] = 1
        state = np.concatenate([cache_decFile_req_state, self.current_state['beta']])
        return state
 
    def calculate_beta(self):
        '''calculate historgram of cached files i.e. beta'''
        beta = np.sum(self.caches, axis=1)/(self.K)#*self.M)
        return beta

    def user_set(self,user):
        self.current_state['user'] = user 
        self.current_state['cache'] = self.caches[:,user]    
    
    def set_TX_file_set(self, TX_file_set):
        ''' set user state based on file request, decoding and cadhing status 
            '''
        self.current_state['TX_file_set']=TX_file_set

                                                   
#%%        
    def find_largest_clique(self,requests, caches): # Find largest clique
        ''' helper function that finds largest clique'''
        
        # discard users without any request
        aux = np.where(requests != NULL())[0]
        requests = requests[aux]
        caches = caches[:, aux]
        usrs_with_req = aux
        n_left = len(aux)   # number of users remaining
        
        max_clique_len = 0
        shuffled_indices = [i for i in range(n_left)]
        shuffle(shuffled_indices)
        for k in range(n_left):     # loop over users i
            i = shuffled_indices[k]
            req_i = requests[i]     # request from current user
            usr_candidates = []     # This will store the candidates to form a clique with current user
            
            for kk in range(k+1, caches.shape[1]):       # Loop over users later than i
                j = shuffled_indices[kk]
                req_j = requests[j]
                if caches[req_i, j] == 1 and caches[req_j, i] == 1:   # Do nothing unless user j stores or demands req_i
                    usr_candidates.append(j)            # ...store as viable candidate
                elif req_i == req_j:               # If current user stores or demands req_j... 
                    usr_candidates.append(j)            # ...store as viable candidate
                    
            # Find largest clique that includes current user i
            clique_i= []
            if len(usr_candidates) == 0:
                clique_i.append(usrs_with_req[i])
            elif max_clique_len >= 1 + len(usr_candidates):
                clique_i.append(0)       # I already have a clique larger than all the candidates
            else:
                aux = self.find_largest_clique(requests[usr_candidates], caches[:, usr_candidates])
                aux = [usr_candidates[i] for i in aux]
                clique_i = usrs_with_req[np.concatenate(([i], aux))]
            
            # If current clique is largest so far, store it in usr_clique (output)
            if max_clique_len < len(clique_i):
                usr_clique = clique_i
                max_clique_len = len(clique_i)
                
        return usr_clique
    
    def convert_1hot_to_numerical_caches(self):
        caches_num = np.zeros(shape=[self.M,self.K],dtype=(int))
        for k in range(self.K):
            caches_num[:,k]= np.where(self.caches[:,k]==1)[0] + 1 # +1 to index from 1 to M
        return caches_num

    def find_most_wanted_plain_file(self,requests,N=None):
        if N==None: N= self.N+1
        ''' find signgle plain file (not in clique) that is most wanted by different users    '''
        alpha = np.histogram(requests, np.arange(0,N), density=False)
        alpha = alpha[0]#[1:]   # Number of times each file is requested
        max_occurance_plain_files = np.max(alpha)    # max num requests for a file
        args_of_max_occurance = np.where(alpha == max_occurance_plain_files)[0]
        file_max_wanted = args_of_max_occurance[np.random.randint(args_of_max_occurance.size)]    # randomly pick one of the files for which there are a_max requests
        return (file_max_wanted,max_occurance_plain_files) 
    #%%
    def randperm(self, M,K,N,method='numerical'):
        ''' if method is 'numerical' build an MxK random matrix of elements from 1 to N
        else if method='binary' make an NxK random matrix of M ones in each column.
        '''
        if method== 'numerical':
            caches = np.zeros([M,K],dtype=int)
            for k in range(K):
                #permutaion_matrix[:,k] = np.random.permutation(np.arange(1,N+1))[0:M]
                caches[:,k] = np.random.choice(a=np.arange(1,N+1),size=(M), replace=False)
        elif method=='binary':
            caches = np.zeros([N,K],dtype=int)
            for k in range(K):                
                caches[:,k] = np.random.choice(a=np.concatenate((np.zeros(N-M,dtype=(int)),np.ones(M,dtype=(int)))),size=(N), replace=False)
        return caches
    

    def generate_requests(self,requests, distribution, Prob_req):
        ''' Generate new requests for already satisfied users that have requests=0
        '''
        requests = np.array(requests)
        index_zero = np.where(requests== NULL())
        size_of_zeros = index_zero[0].size
        new_request_filter = np.random.rand(size_of_zeros)> Prob_req
        ind_new_requests = index_zero[0][new_request_filter]
        #requests[ind_new_requests] = np.random.randint(low=1, high=self.N, size=ind_new_requests.size)        
        requests[ind_new_requests] = self.randi_pmf(distribution, ind_new_requests.size)        
        
        return requests, ind_new_requests

    def randi_pmf(self,pmf,length):
        '''generate an array of samples with len = lengthn with distribution pmf'''        
        symbols = np.arange(len(pmf))
        return np.random.choice(symbols, size=[length], p=pmf)
    
    def generate_distribution_samples(self,param,disribution_type='zipf'):
        '''generate request distribution for the given parameters'''
        if disribution_type == 'zipf':
            distribution = [i**(-param) for i in range(1,1+self.N)]
        return distribution/np.sum(distribution)
    #%%
    
    
    def find_decoding_candidates(self,TX_file,TX_file_set): 
        '''finds cadidate users that can decode files from transmitted file'''
        candidates = np.where(self.caches[TX_file,:] == 0)[0]  # candidates to cache file_rx (they do not have it yet)
        if len(TX_file_set) > 1:
            for otherFile in TX_file_set[np.where(TX_file_set != TX_file)]:
                candidates = candidates[np.where(self.caches[otherFile, candidates] == 1)[0]]  # keep only candidates caching file_xor        
                    
        return candidates
    
    def can_user_decode_file(self, TX_file, TX_file_set):
        '''check if user can decode file TX_file from TX-file_set based on its cache'''
        user = self.current_state['user']
        # if self.caches[TX_file, user]:
        #     return False #already in cache no need to decode
        can_decode = False
        
        files = np.unique(TX_file_set)
        other_files = files[np.where(files != TX_file)]
        can_decode = all(self.caches[other_files, user]) # if user has all other files in cache?
        return can_decode
        
    #%%
    def update_cache(self, user, TX_file,  gamma): 
        # for each user implement the strategy and update beta and cache of the env
        # Loop over users who have been able to decode the file

        K = self.K
        N = self.N
        requests = self.requests
        beta = np.array(self.current_state['beta'])
               
        metric = np.zeros(N)
        for file in range(N):  # Loop over files in the user's cache
            if self.caches[file,user]==1:
                metric[file] = self.compute_value(self.caches[:, user], file, beta, requests[user], gamma)
        metric_new = self.compute_value(self.caches[:, user], TX_file, beta, requests[user], gamma)

        if np.where(metric != 0)[0].size == 0:#some times all metrics tend to zero (when zipf param is large)
            pos = np.random.randint(N) #take a random file 
        else:
            nonzero_ind = np.where(metric != 0)[0]
            pos = nonzero_ind[np.argmin(metric[nonzero_ind])]

        val = metric[pos]

        if val < metric_new:  # If the new file is better than the worst in the cache, replace it
            # self.current_state['beta'][pos] -= 1  # update beta
            # self.current_state['beta'][TX_file] += 1
            #self.update_beta(pos,TX_file)
            self.caches[TX_file, user] = 1
            self.caches[pos, user] = 0
            
            self.replaced_flag = 1
      
    def update_beta(self):
        self.current_state['beta']=self.calculate_beta()

    def  compute_value(self,cache, file, beta, request, gamma):       
        #formula 3 and 5 of the paper

        
        if request!= NULL():
            value = np.multiply(gamma[file]*beta[request],1-beta[file])
        else:
            value = 1 - gamma[file]*(1-beta[file])
                    
            # Add the sum of the cache term
            aux = 1-np.array(beta*cache)
            aux *=gamma*cache*beta[file]
            value = value - np.sum(aux)
        
            # Add the sum of the non-cache term
            aux = gamma* np.array(beta)*(1-beta[file])
            value = value + np.sum(aux) - np.sum(aux*cache)
        
            # Subtract file i term
            if cache[file]==1:
                value += gamma[file]*beta[file]*(1-beta[file])
            else:
                value -= gamma[file]*beta[file]*(1-beta[file])
                   
            # Multiply everything by gamma(file)
            value *= gamma[file]
        return value

#%%
if __name__ == '__main__':
    """Main function
        Main function to test the env code and show an example.
    """
    #np.random.seed(10)
    env = CacheEnv(cache_size=50, number_of_users = 30, number_of_files_in_server=100, req_pdf_param=4)
    print("Reseting Env...")
    env.reset()
    # print("Go DOWN...")
    # env.step(1)

    
    M= env.M
    N= env.N
    K= env.K
    #caches = env.randperm(M,K,N,method='onehot')
   
    #requests = np.random.permutation(np.arange(N))[0:K]
    requests = env.requests
    # print(requests)
    zipf = env.generate_distribution_samples(param=3)
    fig = plt.plot(zipf)
    #fig.show()
    requests,ind = env.generate_requests(requests,zipf,0.2)
    print(f'requests:\n {requests} \n indices:{ind}')
    # hist = np.histogram(caches, bins=np.arange(1,N+2), density=False)
    # print(hist)
    # print(sum(hist[0]))
    
    # test = np.random.choice(a=np.arange(1,11),size=(5), replace=False)
    # cache_onehot = np.zeros([1,10],dtype=int)
    # cache_onehot[0][test] = 1
    # print(cache_onehot[0])
     

    print('caches')
    print(env.caches)
    #print(env.numerical_caches)
    
    cache_num=env.convert_1hot_to_numerical_caches()
 
    usr_clique = env.find_largest_clique(requests, env.caches)
    print(f'largest clique:{usr_clique}')
    print(f'numerical caches:\n{env.caches}')
    beta = env.calculate_beta()
    print(f'beta:\n{beta}')
    
    print('find plain files hist:')
    #requests=[1,0,1,2,0,1,4,1,0,3,2,4,0,5,2,5,5,5]
    alpha = env.find_most_wanted_plain_file(requests)
    print(alpha)
    print(requests)
    
    
   
    