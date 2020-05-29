###########################################Author : Ye Min Oo ( McE ) ###############################################################
import numpy as np
import matplotlib.pyplot as plt

class Dense():
        def __init__(self, layer, n_inputs, n_neurons, activation):
            self.layer = layer
            self.weights = np.random.randn(n_neurons, n_inputs)
            self.bias = np.zeros((n_neurons, 1)) 
            self.activation = activation
        def params_packing(self, w,b):
            WL["layer" + str(self.layer)] = w 
            bL["layer" + str(self.layer)] = b
