from Classes.Network import Network
from Classes.Buffer import Buffer
from Classes.disc_pendulum import DPendulum
import numpy as np
from numpy.random import randint
import config
import time
import matplotlib.pyplot as plt
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
                x_next, r = self.dpendulum.step([u_idx]) # it updates also x
    
                # store the transition in the buffer
                self.buffer.store_experience(x, u_idx, r, x_next)

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
            
            # append the total reward to the history
            total_reward_history.append(reward)
            
            # decrease the exploration probability
            self.epsilon = np.exp(-config.EXPL0RATION_DECREASING_DECAY*nep)
            self.epsilon = max(self.epsilon, config.EXPLORATION_MIN_PROB)
            
            if i % 10 == 0:
                print("Evaluate Q")
                self.evaluate_Q()
            self.NN.Q.save_weights("MODELS/test.h5")

    
  # get greedy input for the Q function        
    def get_input_greedy_Q(self, x):
        # get the action according to the epsilon greedy policy
        if np.random.rand() < self.epsilon:
            u_index = np.random.randint(0, config.dnu)
            input_max = self.dpendulum.d2cu(u_index)
        else:
            # get the action according to the Q function 
            # compute the best Q value according to the x and u given as input
            u_index = 0 # initialize the best u as the first one
            Q_value_max = 0 # initialize the best Q value as the first one
            input_max = 0 # initialize the best input as the first one
            # for i in range(config.dnu):
            #     # compute the "continuous" input
            #     input = self.dpendulum.d2cu(i)
            #     # concatenate the state and the input: get_critic needs 3 (state and input) rows and 1 column
            #     xu = np.reshape(np.append(x, input), (self.dpendulum.pendulum.nx+1,config.actuator_dim))
            #     # convert the input to a tensor
            #     xu = self.NN.np2tf(xu)
            #     # compute the Q value from the Q_ function
            #     Q_value = self.NN.Q(xu)
            #     # convert the tensor to a numpy value 
            #     Q_value = self.NN.tf2np(Q_value)
            #     if Q_value > Q_value_max:
            #         Q_value_max = Q_value
            #         u_index = i
            #         input_max = input
            xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
            # xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),self.dpendulum.u_values)],(config.state_dim+1,self.dpendulum.dnu))
            u_index = np.argmax(self.NN.Q(xu.T))
            input_max = self.dpendulum.u_values[u_index]
            
        return u_index, input_max

    # get greedy input for the Q_target function
    def get_input_greedy_Q_target(self, x):
        # get the action according to the Q_target function 
        # compute the best Q_target value according to the x and u given as input
        u_index = 0 # initialize the best u as the first one
        Q_target_value_max = 0 # initialize the best Q_target value as the first one
        input_max = 0 # initialize the best input as the first one
        # for i in range(config.dnu):
        #     # compute the "continuous" input
        #     input = self.dpendulum.d2cu(i)
        #     # concatenate the state and the input
        #     xu = np.reshape(np.append(x, input), (self.dpendulum.pendulum.nx+1,config.actuator_dim))
        #     # convert the input to a tensor
        #     xu = self.NN.np2tf(xu)
        #     # compute the Q_target value from the Q_target function
        #     Q_target_value = self.NN.Q_target(xu) 
        #        # convert the tensor to a numpy value 
        #     Q_target_value = self.NN.tf2np(Q_target_value)

        #     if Q_target_value > Q_target_value_max:
        #         Q_target_value_max = Q_target_value
        #         u_index = i
        #         input_max = input
        x = np.reshape(x,(config.state_dim,1))
        xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
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
        for i in range(config.MINI_BATCH_SIZE):
            _,u_next[i] = self.get_input_greedy_Q_target(x_next[i,:])
        
        xu_next = np.concatenate([x_next,u_next],axis=1).T
  
        # convert the inputs to tensors
        xu = self.NN.np2tf(xu)
        xu_next = self.NN.np2tf(xu_next)
        r = self.NN.np2tf(r)
        # update the Q function using a set of experiences from the replay buffer
        self.NN.update(xu, r, xu_next)
  
  
    def evaluate_Q(self):
        '''Roll-out from random state using greedy policy.'''
        self.dpendulum.reset()
        x = x0 = self.dpendulum.x
        reward = 0.0
        gamma_i = 1
        for i in range(config.LENGTH_EPISODE):
            # get the action according to the Q function 
            # compute the best Q value according to the x and u given as input
            u_index = 0 # initialize the best u as the first one
            Q_value_max = 0 # initialize the best Q value as the first one
            # for i in range(config.dnu):
            #     # compute the "continuous" input
            #     input = self.dpendulum.d2cu(i)
            #     # concatenate the state and the input: get_critic needs 3 (state and input) rows and 1 column
            #     xu = np.reshape(np.append(x, input), (self.dpendulum.pendulum.nx+1,config.actuator_dim))
            #     # convert the input to a tensor
            #     xu = self.NN.np2tf(xu)
            #     # compute the Q value from the Q_ function
            #     Q_value = self.NN.Q(xu)
            #     # convert the tensor to a numpy value 
            #     Q_value = self.NN.tf2np(Q_value)
            #     if Q_value > Q_value_max:
            #         Q_value_max = Q_value
            #         u_index = i
            xu = np.reshape([np.append([x]*np.ones(self.dpendulum.dnu),np.arange(self.dpendulum.dnu))],(config.state_dim+1,self.dpendulum.dnu))
            u_index = np.argmax(self.NN.Q(xu.T))
                    
            x, r = self.dpendulum.step([u_index]) # it updates also x
   
            reward += gamma_i*r
            gamma_i *= config.DISCOUNT
            self.dpendulum.render()
            
        print("Real cost to go of state", x0[0], x0[1], ":", reward)
        print("Final state:", x[0], x[1])
        