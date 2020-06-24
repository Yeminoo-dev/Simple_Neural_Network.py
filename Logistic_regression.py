import numpy as np 

values = {}
params = {}
derivatives = {}
class linear_reg:
	def __init__(self,X,Y):
		values['X'] = X
		values['Y'] = Y 
		params['W'] = np.random.rand(1,X.shape[0])
		params['b'] = 0
	def sigmoid(self,x):
		return 1/(1+np.exp(-x))
	def d_sigmoid(self,x):
		return self.sigmoid(x)*(1-self.sigmoid(x))
	def h(self):
		return self.sigmoid(np.dot(params['W'],values['X']) + params['b'])
	def loss(self):#sum across examples
	    m = values['Y'].shape[1]
	    return 0.5*np.sum((values['Y'] - self.h())**2)/m
	def derivatives(self):
		m = values['Y'].shape[1]
		derivatives['dW'] = (-1/m)*np.sum(self.d_sigmoid(self.h())*(values['Y'] - self.h())@values['X'].T)
		derivatives['db'] = (-1/m)*np.sum(self.d_sigmoid(self.h())*(values['Y'] - self.h()))
	def fit(self,learning_rate,epochs):
		for i in range(epochs):
			self.h()
			self.derivatives()
			params['W'] -= learning_rate*derivatives['dW']
			params['b'] -= learning_rate*derivatives['db']
			if i % 100 == 0:
				print("Loss after " + str(i) + " epochs : ", self.loss())
	def predict(self,x):
		c = self.sigmoid(np.dot(params['W'],x) + params['b'])
		if c >= 0.5:
			print("Output : 1")
		else:
			print("Output : 0")

inputs = np.array([[0,1,0],
	               [1,0,0]])
outputs = np.array([[0,1,0]])
reg = linear_reg(inputs,outputs)
reg.fit(0.01,10000)
reg.predict(np.array([[0],
	                  [0]]))
