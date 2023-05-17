from Network import Network
from Buffer import Buffer
from disc_pendulum import DPendulum
import numpy as np
from numpy.random import randint
import config
class DQN:
	'''
	This class implements the DQN algorithm, which is used to train the Q function.
	'''
	def __init__(self):
		# create the pendulum object
		self.pendulum = DPendulum(config.nbJoint, config.dnu, config.vMax, config.uMax)
		# create the Q network
		self.NN = Network(config.state_dim, config.actuator_dim, config.DISCOUNT, config.QVALUE_LEARNING_RATE)
		# create the replay buffer
		self.buffer = Buffer(config.BUFFER_SIZE, config.MINI_BATCH_SIZE)
	
		# parameters for the algorithm
		self.epsilon = 1.0
  
		
	
	def algorithm(self):
		# run the algorithm
		for i in range(config.NUM_EPISODE):
			x = self.pendulum.reset()
			
			for j in range(config.LENGTH_EPISODE):
        
          
	def get_input_greedy(self, x):
		# get the action according to the epsilon greedy policy
		if np.random.rand() < self.epsilon:
			u_index = np.random.randint(0, config.dnu)
		else:
			# get the action according to the Q function 
			# compute the best Q value according to the x and u given as input
			u_index = 0 # initialize the best u as the first one
			Q_value_max = 0 # initialize the best Q value as the first one
			for i in range(config.dnu):
				# compute the "continuous" input
				input = self.pendulum.d2cu(i)
				# concatenate the state and the input
				xu = np.append(x, input)
				Q_value = self.NN.Q(xu) # or xu.T? 
				if Q_value > Q_value_max:
					Q_value_max = Q_value
					u_index = i
     
		return u_index
