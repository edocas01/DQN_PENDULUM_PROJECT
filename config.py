import os

# Configuration file for the DQN algorithm
TYPE_PENDULUM = 0 # 0: simple pendulum, 1: double pendulum


# Single pendulum parameters
if TYPE_PENDULUM == 0:
    # Environment parameters
    
    nbJoint = 1 	# number of joints
    dnu = 35 		# discretization of the torque
    vMax = 5.0 		# max velocity
    uMax = 5 		# max torque
    state_dim = 2 	# state dimension

    # DQN parameters
    DISCOUNT = 0.99 			            # discount factor
    QVALUE_LEARNING_RATE = 0.001            # learning rate
    BUFFER_SIZE =   20000    		            # replay buffer size
    MINI_BATCH_SIZE = 64		            # mini batch size
    MIN_EXPERIENCE_BUFFER_SIZE = 200        # minimum experience buffer size
    
    NUM_EPISODE = 120			            # number of episodes
    LENGTH_EPISODE = 100 		            # length of episode
    C_UPDATE = 200							# number of steps before updating the target network
    
    EXPLORATION_MIN_PROB = 0.001            # minimum exploration probability
    EXPL0RATION_DECREASING_DECAY = 0.02    # exploration decreasing decay
    
    # string to save files
    string = "MODELS/SPENDULUM/"

else:
    # Double pendulum parameters
    nbJoint = 2 	# number of joints
 
    # string to save files
    string = "MODELS/DPENDULUM/"


# General parameters
actuator_dim = 1 	# actuator dimension

# FLag to train the network or to evaluate it
training = True

# path to save the network and the log
string += "TEST_03_cost"
if not os.path.exists(string):
    # If it doesn't exist, create it
    os.makedirs(string)
save_model = string + '/model.h5' 	# path to save the network
save_log = string + '/log.txt' 		# path to save the log
save_reward = string + '/reward.dat' # path to save the reward

if training == True:
    # save the main parameters to the log
    with open(save_log,"w") as file:
        file.write("--> PENDULUM PARAMETERS <--\n")
        file.write("TYPE_PENDULUM = " + str(TYPE_PENDULUM) + "\n")
        file.write("state_dim = " + str(state_dim) + "\n")
        file.write("nbJoint = " + str(nbJoint) + "\n")
        file.write("dnu = " + str(dnu) + "\n")
        file.write("vMax = " + str(vMax) + "\n")
        file.write("uMax = " + str(uMax) + "\n")
        file.write("actuator_dim = " + str(actuator_dim) + "\n")
        file.write("\n--> NETWORK PARAMETERS <--\n")
        file.write("DISCOUNT = " + str(DISCOUNT) + "\n")
        file.write("QVALUE_LEARNING_RATE = " + str(QVALUE_LEARNING_RATE) + "\n")
        file.write("BUFFER_SIZE = " + str(BUFFER_SIZE) + "\n")
        file.write("MINI_BATCH_SIZE = " + str(MINI_BATCH_SIZE) + "\n")
        file.write("MIN_EXPERIENCE_BUFFER_SIZE = " + str(MIN_EXPERIENCE_BUFFER_SIZE) + "\n")
        file.write("NUM_EPISODE = " + str(NUM_EPISODE) + "\n")
        file.write("LENGTH_EPISODE = " + str(LENGTH_EPISODE) + "\n")
        file.write("C_UPDATE = " + str(C_UPDATE) + "\n")
        file.write("EXPLORATION_MIN_PROB = " + str(EXPLORATION_MIN_PROB) + "\n")
        file.write("EXPL0RATION_DECREASING_DECAY = " + str(EXPL0RATION_DECREASING_DECAY) + "\n")
        file.write("\n--> SAVE LOG <--\n")
