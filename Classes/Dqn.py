from Classes.Network import Network
from Classes.Buffer import Buffer
from Classes.disc_pendulum import DPendulum
import numpy as np
from numpy.random import randint
import config
import time
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
		# initialize gamma
		gamma = 1.0
		# initialize the total training time
		total_training_time = 0
		# run the algorithm
		for i in range(config.NUM_EPISODE):
			# initialize reward for the episode
			reward = 0
			# increase nep
			nep += 1
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
			print("Episode: ", i, " Reward: ", reward, " Time: ", time, " Epsilon: ", self.epsilon)
   
			# append the total reward to the history
			total_reward_history.append(reward)

			# decrease the exploration probability
			self.epsilon = np.exp(-config.EXPL0RATION_DECREASING_DECAY*nep)
			self.epsilon = max(config.EXPLORATION_PROB, config.EXPLORATION_MIN_PROB)
			
	
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
			for i in range(config.dnu):
				# compute the "continuous" input
				input = self.dpendulum.d2cu(i)
				# concatenate the state and the input: get_critic needs 3 rows and 1 column
				xu = np.reshape(np.append(x, input), (self.dpendulum.pendulum.nx+1,self.dpendulum.dnu))
				print(xu)
				# convert the input to a tensor
				xu = self.NN.np2tf(xu)
				Q_value = self.NN.Q.predict(xu) # or xu.T? 
				print(Q_value)
				if Q_value > Q_value_max:
					Q_value_max = Q_value
					u_index = i
					input_max = input
     
		return u_index, input_max

	# get greedy input for the Q_target function
	def get_input_greedy_Q_target(self, x):
		# get the action according to the Q_target function 
		# compute the best Q_target value according to the x and u given as input
		u_index = 0 # initialize the best u as the first one
		Q_target_value_max = 0 # initialize the best Q_target value as the first one
		input_max = 0 # initialize the best input as the first one
		for i in range(config.dnu):
			# compute the "continuous" input
			input = self.dpendulum.d2cu(i)
			# concatenate the state and the input
			# xu = np.append(x, input)
			xu = np.reshape(np.append(x, input), (self.dpendulum.pendulum.nx+1,1))
			print(xu)
			# convert the input to a tensor
			xu = self.NN.np2tf(xu) # or xu.T? 
			Q_target_value = self.NN.Q_target(xu) 
			print(Q_target_value)
			if Q_target_value > Q_target_value_max:
				Q_target_value_max = Q_target_value
				u_index = i
				input_max = input
     
		return u_index, input_max

	# update the Q function using the mini batch
	def update_Q(self, mini_batch):
		dim_x = config.state_dim
		dim_u = config.actuator_dim
		# extract the data from the mini batch
		# mini_batch = np.asarray(mini_batch, dtype = object)
		# x_batch = mini_batch[:,0:1]
		# u_batch = mini_batch[:,1:1+dim_u]
		# r_batch = mini_batch[:,1+dim_u:1+dim_u+1]
		# x_batch_next = mini_batch[:,1+dim_u+1:]
		x_batch, u_batch, cost_batch, x_batch_next = list(zip(*mini_batch))
		u_batch_next = np.empty(config.MINI_BATCH_SIZE)

		x_batch = np.concatenate([x_batch],axis=1).T
		u_batch = np.asarray(u_batch)
		r_batch = np.asarray(cost_batch)
		# compute the max u' according to the Q_target function for each x' in the mini batch
		# u_batch_next = np.zeros((config.MINI_BATCH_SIZE, config.actuator_dim))
		for i in range(config.MINI_BATCH_SIZE):
			_,u_batch_next[i] = self.get_input_greedy_Q_target(x_batch_next[i])
		
		# create the inputs for the Q function
		xu = (np.append(x_batch, u_batch, axis=1)).T
		# create the inputs for the Q_target function
		xu_next = (np.append(x_batch_next, u_batch_next, axis=1)).T

		# convert the inputs to tensors
		xu = self.NN.np2tf(xu)
		xu_next = self.NN.np2tf(xu_next)

		# update the Q function
		self.NN.update(xu, r_batch, xu_next)
		