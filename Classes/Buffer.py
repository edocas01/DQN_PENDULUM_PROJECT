import numpy as np
from collections import deque
import random

class Buffer:
  '''
  This class implements the replay buffer used to store the transitions (x,u,r,x') during the training.
  This allows to decorrelate the samples and thus to stabilize the training.
  '''
  def __init__(self, buffer_size, mini_bach_size):
    # the buffer size is used to choose when to start the sampling
    self.buffer_size = buffer_size
    # the mini batch size is used to choose the number of samples to use for the training
    self.mini_bach_size = mini_bach_size
    # deque is a list-like container with fast appends and pops on either end
    self.buffer = deque(maxlen=buffer_size)
  
  # insert a transition in the buffer
  def store_experience(self, x, u, r, x_nest):
    self.buffer.append((x, u, r, x_nest))
    
  def sample_mini_batch(self):
    # sample a random mini batch of transitions
    mini_batch = random.sample(self.buffer, self.mini_bach_size)
    return mini_batch