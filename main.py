# import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import os

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
    save_figures = config.string + '/FIGURES/'
    if not os.path.exists(save_figures):
        # If it doesn't exist, create it
        os.makedirs(save_figures)
        
    # load the model
    DQN_Agent.NN.Q.load_weights(config.save_model)
    if config.TYPE_PENDULUM == 0:
        C,X,U = DQN_Agent.evaluate_Q([math.pi,0])
    else:
        # C,X,U = DQN_Agent.evaluate_Q([math.pi,0,0,0])
        C,X,U = DQN_Agent.evaluate_Q()
    
    # plot the results
    time = np.arange(0,config.LENGTH_EPISODE)   
    plt.figure()
    plt.plot(time, C[:], "k")
    plt.gca().set_xlabel('Iteration')
    plt.gca().set_ylabel('Reward')
    string = "Reward over an episode"
    plt.title(string)
    plt.plot(time, np.zeros(config.LENGTH_EPISODE), "r--",linewidth=1.0)
    plt.savefig(save_figures + string + '.png')
    
    plt.figure()
    if config.TYPE_PENDULUM == 0:
        plt.plot(time, X[0,:], "k")
        plt.gca().set_xlabel('Iteration')
        plt.gca().set_ylabel('Joint angle [rad]')
        string = "Joint position over an episode"
        plt.title(string)
        plt.plot(time, np.zeros(config.LENGTH_EPISODE), "r--",linewidth=1.0)
        plt.savefig(save_figures + string + '.png')
        
        plt.figure()
        plt.plot(time, X[1,:], "k")
        plt.gca().set_xlabel('Iteration')
        plt.gca().set_ylabel('Joint velocity [rad/s]')
        string = "Joint velocity over an episode"
        plt.title(string)
        plt.plot(time, np.zeros(config.LENGTH_EPISODE), "r--",linewidth=1.0)
        plt.savefig(save_figures + string + '.png')
    else:
        plt.plot(time, X[0,:], "k")
        plt.plot(time, X[1,:], "b")
        plt.gca().set_xlabel('Iteration')
        plt.gca().set_ylabel('Joint angle [rad]')
        plt.legend(['q1','q2'])
        string = "Joint poistions over an episode"
        plt.title(string)
        plt.plot(time, np.zeros(config.LENGTH_EPISODE), "r--",linewidth=1.0)
        plt.savefig(save_figures + string + '.png')
        
        plt.figure()
        plt.plot(time, X[1,:], "k")
        plt.plot(time, X[3,:], "b")
        plt.gca().set_xlabel('Iteration')
        plt.gca().set_ylabel('Joint velocities [rad/s]')
        plt.legend(['dq1','dq2'])
        string = "Joint velocities over an episode"
        plt.title(string)
        plt.plot(time, np.zeros(config.LENGTH_EPISODE), "r--",linewidth=1.0)
        plt.savefig(save_figures + string + '.png')
    
    plt.figure()
    plt.plot(time, U[:], "k")
    plt.gca().set_xlabel('Iteration')
    plt.gca().set_ylabel('Joint torque [Nm]')
    string = "Joint torque over an episode"
    plt.title(string)
    plt.plot(time, np.zeros(config.LENGTH_EPISODE), "r--",linewidth=1.0)
    plt.savefig(save_figures + string + '.png')
    
    R = np.fromfile(config.save_reward, dtype=float)
    time = np.arange(0,config.NUM_EPISODE)
    plt.figure()
    plt.plot(time, R[:], "k")
    plt.gca().set_xlabel('Episodes')
    plt.gca().set_ylabel('Reward')
    string = "Reward over the episodes"
    plt.title(string)
    plt.plot(time, np.zeros(config.NUM_EPISODE), "r--",linewidth=1.0)
    plt.savefig(save_figures + string + '.png')
    
    plt.figure() 
    plt.gca().set_xlabel('Episodes')
    plt.gca().set_ylabel('Average reward')
    string = "Average reward over the episodes"
    plt.title(string)
    if config.TYPE_PENDULUM == 0:
        window = 10
    else:
        window = 20
    av = np.convolve(R, np.ones(window), 'valid') / window
    plt.plot(np.arange(0,len(av)),  av, "k")
    plt.plot(time, np.zeros(config.NUM_EPISODE), "r--",linewidth=1.0)
    plt.savefig(save_figures + string + '.png')
    
    if config.TYPE_PENDULUM == 0:
        V, P, q, dq = DQN_Agent.compute_V_pi()
        DQN_Agent.dpendulum.plot_V_table(V, q, dq, save_figures)
        DQN_Agent.dpendulum.plot_policy(P, q, dq, save_figures)
        
    plt.show()

