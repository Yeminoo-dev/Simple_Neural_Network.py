import numpy as np
###########################################Author : Ye Min Oo ( McE ) ###############################################################
class Dense():
  def __init__(self,layer,n_inputs,n_neurons,activation):
    self.layer = layer
    self.weights = np.random.randn(n_neurons, n_inputs)
