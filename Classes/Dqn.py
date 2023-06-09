from Classes.Network import Network
from Classes.Buffer import Buffer
from Classes.disc_pendulum import DPendulum
import numpy as np
from numpy.random import randint
import config
import time
import matplotlib.pyplot as plt
import math
class DQN:
    '''
    This class implements the DQN algorithm, which is used to train the Q function.
    '''
    def __init__(self):
        # create the pendulum object
        self.dpendulum = DPendulum(config.nbJoint, config.dnu, config.vMax, config.uMax)
        # create the Q network
        self.NN = Network(config.state_dim, config.actuator_dim, config.DISCOUNT, config.QVALUE_LEARNING_RATE)
        # create the replay buffer
        self.buffer = Buffer(config.BUFFER_SIZE, config.MINI_BATCH_SIZE)
    
        # parameters for the algorithm
        self.epsilon = 1.0
        

    def algorithm(self):
        # initialize C
        C = 0
        # initialize nep
        nep = 0
        # initialize the total reward history
        total_reward_history = []
        
        # initialize the total training time
        total_training_time = 0
        # run the algorithm
        for i in range(config.NUM_EPISODE):
            # initialize reward for the episode
            reward = 0
            # increase nep
            nep += 1
            # initialize gamma
            gamma = 1.0
            # initialize the time for the episode
            time_passed = time.time()
   
            # reset the state of the pendulum
            self.dpendulum.reset()
   
            for j in range(config.LENGTH_EPISODE):
                # increment C
                C += 1
                # get the state of the pendulum
                x = self.dpendulum.x
            
                # get the action according to the epsilon greedy policy
                u_idx, u = self.get_input_greedy_Q(x)
                # apply the action to the pendulum
                if config.TYPE_PENDULUM == 0:
                    x_next, r = self.dpendulum.step([u_idx]) # it updates also x
                else:
                    x_next, r = self.dpendulum.step([u_idx, (self.dpendulum.dnu-1)/2])
                # store the transition in the buffer
                # self.buffer.store_experience(x, u_idx, r, x_next)
                self.buffer.store_experience(x, u, r, x_next)

                # sample from the batch if it is big enough
                if len(self.buffer.buffer) > config.MIN_EXPERIENCE_BUFFER_SIZE:
                    mini_batch = self.buffer.sample_mini_batch()
                    # train the network
                    self.update_Q(mini_batch)

                # update Q_target every C steps
                if C % config.C_UPDATE == 0:
                    self.NN.Q_target.set_weights(self.NN.Q.get_weights())

                # update the reward
                reward += gamma*r
                # update gamma
                gamma *= config.DISCOUNT

            # compute the time for the episode
            time_passed = round(time.time() - time_passed,3)
            # update the total training time
            total_training_time += time_passed
            
            # print the results
            print("Episode: ", i, " Reward: {:.4g}".format(reward), " Time: ", time_passed, " Epsilon: {:.4g}".format(self.epsilon))
            with open(config.save_log, "a") as myfile:
                print("Episode: ", i, " Reward: {:.4g}".format(reward), " Time: ", time_passed, " Epsilon: {:.4g}".format(self.epsilon), file=myfile)
                
            # append the total reward to the history
            total_reward_history.append(reward)
            
            # decrease the exploration probability
            self.epsilon = np.exp(-config.EXPL0RATION_DECREASING_DECAY*nep)
            self.epsilon = max(self.epsilon, config.EXPLORATION_MIN_PROB)

            if i % 30 == 0 and i > 0:
                print("Evaluate Q")
                self.evaluate_Q()
            self.NN.Q.save_weights(config.save_model)
        
        # print the total training time
        print("Total training time: ", total_training_time, " seconds (", total_training_time/60, " minutes)")
        with open(config.save_log, "a") as myfile:
            print("\nTotal training time: ", total_training_time, " seconds (", total_training_time/60, " minutes)", file=myfile)
          
        # save the total reward history
        reward_history_array = np.asarray(total_reward_history)
        reward_history_array.tofile(config.save_reward)  
          
  # get greedy input for the Q function        
    def get_input_greedy_Q(self, x):
        # get the action according to the epsilon greedy policy
        if np.random.rand() < self.epsilon:
            u_index = np.random.randint(0, config.dnu)
            input_max = self.dpendulum.d2cu(u_index)
        else:
            
            x = np.reshape(x,(config.state_dim,1))
            # get the action according to the Q function 
            # xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
            xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),self.dpendulum.u_values)],(config.state_dim+1,self.dpendulum.dnu))

            u_index = np.argmax(self.NN.Q(xu.T))
            input_max = self.dpendulum.u_values[u_index]
            
        return u_index, input_max

    # get greedy input for the Q_target function
    def get_input_greedy_Q_target(self, x):
        # get the action according to the Q_target function 
        x = np.reshape(x,(config.state_dim,1))
        # xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
        xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),self.dpendulum.u_values)],(config.state_dim+1,self.dpendulum.dnu))
        u_index = np.argmax(self.NN.Q_target(xu.T))
        input_max = self.dpendulum.u_values[u_index]
     
        return u_index, input_max

    # update the Q function using the mini batch
    def update_Q(self, mini_batch):
        dim_x = config.state_dim
        dim_u = config.actuator_dim
        # extract the data from the mini batch
        mini_batch = np.concatenate([mini_batch], axis=0)
        x = mini_batch[:,0:dim_x]
        u = mini_batch[:,dim_x:dim_x+dim_u]
        r = mini_batch[:,dim_x+dim_u]
        x_next = mini_batch[:,dim_x+dim_u+1:]

        # xu is a matrix with. 3 (or 5) rows and MINI_BATCH_SIZE columns
        xu = np.concatenate([x,u],axis=1).T
  
        # compute the max u' according to the Q_target function for each x' in the mini batch
        u_next = np.zeros((config.MINI_BATCH_SIZE, config.actuator_dim))
        u_next_index = np.zeros((config.MINI_BATCH_SIZE, config.actuator_dim))
        for i in range(config.MINI_BATCH_SIZE):
            u_next_index[i], u_next[i] = self.get_input_greedy_Q_target(x_next[i,:])
        
        xu_next = np.concatenate([x_next,u_next],axis=1).T
        # xu_next = np.concatenate([x_next,u_next_index],axis=1).T
  
        # convert the inputs to tensors
        xu = self.NN.np2tf(xu)
        xu_next = self.NN.np2tf(xu_next)
        r = self.NN.np2tf(r)
        # update the Q function using a set of experiences from the replay buffer
        self.NN.update(xu, r, xu_next)
  
    # run a simulation using trained Q function
    def evaluate_Q(self, x = None):
        '''Roll-out from random state using greedy policy.'''
        if x is None:
            self.dpendulum.reset()
            x = x0 = self.dpendulum.x
        else:
            if config.TYPE_PENDULUM == 0:
                x = x0 = np.array([[x[0]],[x[1]]])
            else:
                x0 = x = np.asarray(x)
            self.dpendulum.reset(x)
        reward = 0.0
        gamma_i = 1
        C_hist = []
        X_hist = []
        U_hist = []
        for i in range(config.LENGTH_EPISODE):
            
            x = np.reshape(x,(config.state_dim,1))
            # get the action according to the Q function 
            # xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
            xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),self.dpendulum.u_values)],(config.state_dim+1,self.dpendulum.dnu))
            u_index = np.argmax(self.NN.Q(xu.T))
            
            if config.TYPE_PENDULUM == 0:       
                x, r = self.dpendulum.step([u_index]) # it updates also x
            else:
                x, r = self.dpendulum.step([u_index, (self.dpendulum.dnu-1)/2])
            reward += gamma_i*r
            gamma_i *= config.DISCOUNT
            self.dpendulum.render()
            
            C_hist.append(r)
            X_hist.append(x)
            U_hist.append(self.dpendulum.u_values[u_index])
            # U_hist.append(u_index)
        
        X_hist = np.reshape(X_hist,(config.LENGTH_EPISODE,config.state_dim))
        X_hist = X_hist.T   
        if config.TYPE_PENDULUM == 0:
            print("Real cost to go of state", x0[0], x0[1], ":", reward)
            print("Final state:", x[0], x[1])
        else:
            print("Real cost to go of state", x0[0], x0[1], x0[2], x0[3],":", reward)
            print("Final state:", x[0], x[1], x0[2], x0[3])
            
        
        return C_hist, X_hist, U_hist
        
    # Compute the value function V and policy pi    
    def compute_V_pi(self):
        definition = 100
        q = np.linspace(-math.pi, math.pi, num=definition)
        dq = np.linspace(-config.vMax, config.vMax, num=definition)
        
        PI = np.zeros(shape=(definition, definition))
        V = np.zeros(shape=(definition, definition))
        
        for i in range(definition):
            for j in range(definition):
                
                x = np.array([[q[i]],[dq[j]]])
                # xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
                xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),self.dpendulum.u_values)],(config.state_dim+1,self.dpendulum.dnu))
                u_index = np.argmax(self.NN.Q(xu.T))
                input_max = self.dpendulum.u_values[u_index]
            
                V[i,j] = np.max(self.NN.Q(xu.T))
                PI[i,j] = input_max
                # PI[i,j] = u_index
                
        return V, PI, q, dq
