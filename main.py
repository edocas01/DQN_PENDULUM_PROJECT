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

# create the DQN agent: it will contain the pendulum, the network and the buffer
DQN_Agent = DQN()

training = True


if training == True:
    # run the algorithm
    DQN_Agent.algorithm()
else:
    # load the model
    DQN_Agent.NN.Q.load_weights("MODELS/test_02.h5")
    DQN_Agent.evaluate_Q()
