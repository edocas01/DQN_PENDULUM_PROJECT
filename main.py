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

# run the algorithm
DQN_Agent.algorithm()



