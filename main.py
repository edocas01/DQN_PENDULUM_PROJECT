# import libraries
import numpy as np
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
    DQN_Agent.evaluate_Q()
