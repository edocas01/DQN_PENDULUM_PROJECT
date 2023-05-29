# Configuration file for the DQN algorithm
TYPE_PENDULUM = 0 # 0: simple pendulum, 1: double pendulum


# Single pendulum parameters
if TYPE_PENDULUM == 0:
    # Environment parameters
    
    nbJoint = 1 	# number of joints
    dnu = 14 		# discretization of the torque
    vMax = 5.0 		# max velocity
    uMax = 5.0 		# max torque
    state_dim = 2 	# state dimension

    # DQN parameters
    DISCOUNT = 0.99 			            # discount factor
    QVALUE_LEARNING_RATE = 0.001            # learning rate
    BUFFER_SIZE =   2000    		            # replay buffer size
    MINI_BATCH_SIZE = 32		            # mini batch size
    MIN_EXPERIENCE_BUFFER_SIZE = 100        # minimum experience buffer size
    
    NUM_EPISODE = 100			            # number of episodes
    LENGTH_EPISODE = 100 		            # length of episode
    C_UPDATE = 250							# number of steps before updating the target network
    
    EXPLORATION_MIN_PROB = 0.001            # minimum exploration probability
    EXPL0RATION_DECREASING_DECAY = 0.03    # exploration decreasing decay

else:
	# Double pendulum parameters
	nbJoint = 2 	# number of joints

actuator_dim = 1 	# actuator dimension
