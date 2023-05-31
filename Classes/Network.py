import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from tensorflow.keras import layers
import numpy as np


class Network:
  '''
  This class implements the neural network used to represent the Q function. 
  The network is composed of 4 fully connected layers with 16, 32, 64 and 64 neurons respectively.
  The activation function is ReLU.
  This NN returns a single output, the Q value (scalar), for each pair of state and action.
  Notice that the state can be a vector of 2 or 4 elements, depending on the pendulum model, while
  the action is a scalar (only first joint actuated).
  '''
  def __init__(self, nx, nu, DISCOUNT, QVALUE_LEARNING_RATE):
    # dimension of the state and action vectors
    self.nx = nx # dimension of the state vector (2 or 4)
    self.nu = nu # dimension of the action vector (1)

    # initialize the Q and Q_target networks
    self.Q = self.get_critic()
    self.Q_target = self.get_critic()
    
    # Set initial weights of targets equal to those of the critic
    self.Q_target.set_weights(self.Q.get_weights())
    
    
    # parameters for the training
    self.DISCOUNT = DISCOUNT
    self.critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
     

  # convert from numpy to tensorflow
  def np2tf(self,y):
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
  # convert from tensorflow to numpy
  def tf2np(self,y):
    return tf.squeeze(y).numpy()
  
  # Create the neural network to represent the Q function
  def get_critic(self):
    inputs = layers.Input(shape=(self.nx+self.nu,)) # leave "," to have as output a tensor of shape (None, 1)
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 
    
    model = tf.keras.Model(inputs, outputs)

    return model
  
  # Update the weights of the Q network using the specified batch of data
  def update(self,xu_batch, cost_batch, xu_next_batch):
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = self.Q_target(xu_next_batch, training=True)   
        # Compute 1-step targets for the critic loss
        y = cost_batch + self.DISCOUNT*target_values                            
        # Compute batch of Values associated to the sampled batch of states
        Q_value = self.Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, self.Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    self.critic_optimizer.apply_gradients(zip(Q_grad, self.Q.trainable_variables))   

    
  