# DQN_PENDULUM_PROJECT
This repository contains the code developed by Edoardo Castioni and Simone Manfredi for the project of the course "Advanced optimization-based robot control" of professor Del Prete. The project consist in the implementation of the DQN algorithm in order to stabilize the inverse pendulum. The whole environment is written in pyhton
![Alt text](untitled.mp4)

## Papers and related works
- [_Implementing the Deep Q-Network_](https://arxiv.org/abs/1711.07478)
- [_Human-level control through deep reinforcement learning_](https://www.nature.com/articles/nature14236)

## Documentation
- [_Keras_](https://keras.io/api/layers/core_layers/input/)
- [_DQN_](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)


## Code Management
The code is mainly divided in classes:
- Pendulum: class for the pendulum model
- DPendulum: class for the discretized version of the pendulum
- Network: class containing the Q and Q_target functions
- Buffer: class for mini batch sampling and experiences storing
- DQN: class for implementing the whole algorithm

In order to run the code it is necessary to properly configure the "config.py" with all the hyper-parameters of the Dqn algorithm. In addiction
it is possible to select the "training" flag to choose between two options:
- perform the training using the Dqn algorithm (and save results in the selected files)
- load the pre-trained model and evaluate the trained Q function

Once everithing is setted, it is sufficient to run the "main.py"



