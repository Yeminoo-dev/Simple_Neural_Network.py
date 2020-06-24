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
	def h(self):
		return np.dot(params['W'],values['X']) + params['b']
	def loss(self):#sum across examples
	    return 0.5*np.sum((values['Y'] - self.h())**2)
	def derivatives(self):
		m = values['Y'].shape[1]
		derivatives['dW'] = (-1/m)*np.sum((values['Y'] - self.h())@values['X'].T)
		derivatives['db'] = (-1/m)*np.sum((values['Y'] - self.h()))
	def fit(self,learning_rate,epochs):
		for i in range(epochs):
			self.h()
			self.derivatives()
			params['W'] -= learning_rate*derivatives['dW']
			params['b'] -= learning_rate*derivatives['db']
			if i % 100 == 0:
				print("Loss after " + str(i) + " epochs : ", self.loss())
	def predict(self,x):
		print(np.dot(params['W'],x) + params['b'])

inputs = np.array([[0,1,2,3,4]])
outputs = np.array([[0,2,4,6,8]])
reg = linear_reg(inputs,outputs)
reg.derivatives()
print(derivatives)
reg.fit(0.01,10000)
reg.predict(10)
