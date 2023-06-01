# import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# import dependencies
from Classes.disc_pendulum import DPendulum
from Classes.Dqn import DQN
from Classes.Buffer import Buffer
from Classes.Network import Network
import config
    
# create the DQN agent: it will contain the pendulum, the network and the buffer
DQN_Agent = DQN()

if config.training == True:
    # run the algorithm
    DQN_Agent.algorithm()
else:
    # load the model
    DQN_Agent.NN.Q.load_weights(config.save_model)
    C,X,U = DQN_Agent.evaluate_Q([math.pi,0])
    # C,X,U = DQN_Agent.evaluate_Q()
    
    # plot the results
    time = np.arange(0,config.LENGTH_EPISODE)   
    plt.figure()
    plt.plot(time, C[:], "k")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Reward')
    plt.title("Reward over an episode")
    
    plt.figure()
    plt.plot(time, X[0,:], "k")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('joint angle [rad]')
    plt.title("Joint poistion over an episode")
    
    plt.figure()
    plt.plot(time, X[1,:], "k")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('joint velocity [rad/s]')
    plt.title("Joint velocity over an episode")
    
    plt.figure()
    plt.plot(time, U[:], "k")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('joint torque [Nm]')
    plt.title("Joint torque over an episode")
    
    R = np.fromfile(config.save_reward, dtype=float)
    time = np.arange(0,config.NUM_EPISODE)
    plt.figure()
    plt.plot(time, R[:], "k")
    plt.gca().set_xlabel('Episodes')
    plt.gca().set_ylabel('Reward')
    plt.title("Reward over the episodes")
    plt.show()
