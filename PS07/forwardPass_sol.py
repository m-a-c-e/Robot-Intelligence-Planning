import numpy as np

def forwardPass(nn, X, outputLayer=None, return_output=False):
	"""
	Authors: 	Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu
				Rohan Paleja <rpaleja3@gatech.edu>
				Letian Chen <letian.chen@gatech.edu>

	Date: 		24 JUN 2020

	This function takes as input a neural netowrk, nn, and inputs, X, and
	performs a forward pass on the neural network. The code assumes that
	layers, {1,2,...,l-1}, have ReLU activations and the final layer has a
	linear activation function. The code ignores bias terms in the neural
	network.

	Inputs:

	nn          -	The weights and biases of the neural network. nn[i][0]
					corresponds to the weights for the ith layer and
					nn[i][1] corresponds to the biases for the ith layer
	X           - 	This matrix is n x d matrix and contains the input
					features for n examples, each with d features.

	Outputs:

	Y           -   This term is an n x 1 vector of predicted labels for
					the n examples.
	"""

	numLayers = len(nn)                                            # get the number of layers of our neural network

	outputs = [None] * numLayers                                   # Initialize the cell to store the outputs of the hidden layers
	lin_outputs = [None] * numLayers                               # Linear outputs of different layers.
	inputs = []													   # Initialize the cell to store the inputs to the hidden layers
	for i in range(numLayers):
		# Compute the input to layer i
		if i == 0:
			# Input layer uses the data, X, X is a list here for the states.
			inputs.append(np.array(X).reshape(6, 1))
		else:
			# Hidden layers use the output of the previous layers
			inputs.append(outputs[i-1])

		if i < numLayers-1:                                         # range function is exclusive so we do numLayers - 1
			# If layer i is not the output layer, then apply the ReLU activation function for the nodes at this layer.
			lin_outputs[i] = nn[i][0]@inputs[i]+nn[i][1]
			outputs[i] = (lin_outputs[i]>0)*lin_outputs[i]
		else:
			# If layer i is the output layer,
			if outputLayer == 'Softmax':
				# Apply softmax
				lin_outputs[i] = nn[i][0]@inputs[i]+nn[i][1]
				p = np.exp(lin_outputs[i])
				outputs[i] = p/np.sum(p)
			else:
				# Apply a linear activation (i.e., no activation)
				lin_outputs[i] = nn[i][0] @ inputs[i] + nn[i][1]
				outputs[i] = nn[i][0] @ inputs[i] + nn[i][1]
	Y_hat = outputs[-1]
	if return_output:
		return Y_hat, outputs, inputs, lin_outputs
	else:
		return Y_hat

