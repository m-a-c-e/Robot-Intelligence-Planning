'''
Authors: 	Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu
			Manisha Natarajan <mnatarajan30@gatech.edu>
			
Date: 		05 JUL 2020
'''
import matplotlib.pyplot as plt
import numpy as np


from createNeuralNet import createNeuralNet
from forwardPass import forwardPass
from pongPhysics import pongPhysics
from plotDiagnostics import plotDiagnostics
from backprop import backprop

np.random.seed(1)  # Set random seed for repeatability
save = True


def computeMSE(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    mse = np.mean(np.square(labels - predictions)) / 2

    return mse


class A2C:
	def __init__(self):

		### DO NOT CHANGE HYPERPARAMETERS FOR THIS ASSIGNMENT ###
		# You are welcome to play with hyperparameters for your own education/understanding
		# Please submit to gradescope with the original parameters

		self.alpha = 2e-1  # Learning rate for all gradient descent methods
		self.alphaQ = 1e-3
		self.alphaV = 1e-4
		self.numNodesPerLayerP = [8]  # Vector of the number of hidden nodes per layer, indexed from input -> output
		self.numNodesPerLayerQ = [8]
		self.numNodesPerLayerV = [8]
		self.K = 32  # Number of samples in our minibatch
		self.I = 20000  # Number of gradient descent iterations
		self.T = 100  # Max time steps for the game (i.e., policy roll-out)
		self.n_Dprime = 100 # size of the dataset used to update the Q nn.
		self.gamma = 0.95  # discount factor
		self.epsilon_o = 0.05  # Minimum amount of noise to maintain in the epsilon-greedy policy rollouts
		self.epsilon_decay_const = 1e-4  # Rate at which epsilon is annealed from 1 to epsilon_O
		self.P = 25  # Number of minibatch gradient descent steps taken per game played
		self.difficultyLevel = 1  # Difficulty level (see below)
		self.numInputDims = 6
		self.numOutputDims = 2  # Dimensionality of the outputs space (i.e., for ``move up'' vs. ``move down'')
		self.numLayers = len(self.numNodesPerLayerP)  # Number of hidden layers
		self.nn = createNeuralNet(self.numNodesPerLayerP, self.numInputDims, self.numOutputDims)  # Initialize player learner
		self.nnQ = createNeuralNet(self.numNodesPerLayerQ, self.numInputDims, self.numOutputDims)
		self.nnV = createNeuralNet(self.numNodesPerLayerQ, self.numInputDims, 1)
		self.L_V = np.zeros((self.I, 1)) # loss for the V network
		self.L_Q = np.zeros((self.I, 1))  # loss for the Q network
		self.L = np.zeros((self.I, 1))  # loss for the policy
		self.winLossRecord = np.zeros((self.I))
		self.data_nn = (self.alpha, self.alphaQ, self.alphaV,self.numNodesPerLayerP, self.numNodesPerLayerQ,self.numNodesPerLayerQ, self.n_Dprime)

	def run_a2c(self):
		# Code to plot
		fig, axes = plt.subplots(1, 3)
		fig.suptitle(r"A2C_plot.png")
		line1, = axes[0].plot([], '-b', linewidth=3)
		line2, = axes[0].plot([], '-r', linewidth=3)
		line3, = axes[0].plot([], '.k', markersize=12)
		axes[0].set_xlim(0, 1)
		axes[0].set_ylim(0, 1)
		fig.canvas.draw()
		axbackground = fig.canvas.copy_from_bbox(axes[0].bbox)
		# plt.show()
		fig.canvas.flush_events()


		D = []
		for i in range(self.I):
			if i % 1000 == 0:
				print("Iteration {}".format(i))
			#################################################################
			#########                 COLLECT DATA                  #########
			#################################################################

			# Initialize the first state of the game. The state consists of the
			# following features.
			# 1) y-position of the player's paddle
			# 2) y-position of the opponent's paddle
			# 3) x-position of the ball
			# 4) y-position of the ball
			# 5) x-velocity of the ball
			# 6) y-velocity of the ball

			s = [0.5, 0.5, 0.5, 0.5, -1, 0]

			# Randomly initialize the y-velocity of the ball and normalize so that
			# the speed of the ball is 1.

			vel_init = [-1, np.random.uniform() - 0.5]
			vel_init = vel_init / np.sqrt(np.dot(vel_init, vel_init))
			s[4:] = vel_init

			# INITIALIZE A LIST TO STORE TRAJECTORY <S,A,R,S'> FOR EACH TIME STEP t in range(T)
			tau = []
			T_terminal = self.T
			for t in range(self.T):

				# Here, we draw actions from a policy (i.e., a probability mass function)
				probs = forwardPass(self.nn, s, 'Softmax')

				a_1 = np.random.choice(2, p=probs.ravel())  # SAMPLE FROM A PROBABILITY DISTRIBUTION DEFINED BY PROBS

				# For now player 2 is not going to take any actions
				a_2 = -1

				# Apply transition function and get our reward. Note: the fourth
				# input is a Boolean to determine whether to plot the game being
				# played.
				PlottingBool = False
				s_prime, r = pongPhysics(s, a_1, a_2, PlottingBool, axes, fig, line1, line2, line3, axbackground)

				# Store the <s,a,r,s'> transition into a trajectory
				tau.append((s, a_1, r[0], s_prime))  # tau[t] = <s,a_1,r[0]>
				D.append((s, a_1, r[0], s_prime))
				# Determine if the new state is a terminal state. If so, then quit
				# the game. If not, step forward into the next state.
				if r[0] == -1 or r[1] == -1:
					# The next state is a terminal state. Therefore, we should
					# record the outcome of the game in winLossRecord for game i.
					self.winLossRecord[i] = float(r[0] == 1)
					T_terminal = t+1
					break

				else:
					# Simply step to the next state
					s = s_prime

			
			# update the policy network
			self.update_policy(tau, T_terminal, i)
			# update the q network
			self.update_q(D, i)
			# update the v network
			self.update_v(tau, T_terminal, i)

			# Plot loss per iter and win-loss record per iter.
			if i % 100 == 0:
				plotDiagnostics(L=self.L, winLossRecord=self.winLossRecord, i=i,axes=axes, fig=fig, data=self.data_nn, method='A2C')
			fig.canvas.flush_events()


	def update_policy(self, tau, T_terminal, i):
		######################################################################
		##############                 UPDATE POLICY            ##############
		######################################################################
		for t in range(T_terminal):
			# get the state, s_t
			s = tau[t][0]

			# get the current action, a_t
			a = tau[t][1]

			# get the probability of taking action a_t in state s_t, i.e. \pi(a_t | s_t)
			probs = forwardPass(self.nn, s, 'Softmax')
			Pr_a_Given_s = probs[a]

			# We update policy based on Q(s,a) - V(s) value instead of estimate of expected future reward
			A_t = forwardPass(self.nnQ, s, 'Linear')[a] - forwardPass(self.nnV, s, 'Linear')
			
			# Compute the gradient for the neural network
			g = backprop(self.nn, s, a, 'Softmax')  # third entry should be the action

			# Update the neural network parameters (normalizing for batchSize).
			for j in range(len(self.nn)):

				self.nn[j][0] = self.nn[j][0] + self.alpha * A_t * (1 / Pr_a_Given_s) * g[j][0]/T_terminal  # Update the neural network weights
				self.nn[j][1] = self.nn[j][1] + self.alpha * A_t * (1 / Pr_a_Given_s) * g[j][1]/T_terminal  # Update the neural network biases

				# Store into L[i] a measure of how much the network is changing.
				loss1 = np.sum(np.abs(self.alpha * A_t * (1 / Pr_a_Given_s) * g[j][0]))
				loss2 = np.sum(np.abs(self.alpha * A_t * (1 / Pr_a_Given_s) * g[j][1]))
				self.L[i] += (loss1 + loss2)

		# Normalize by the number of time steps.
		self.L[i] /= T_terminal

	def update_q(self, D, i):
		######################################################################
		##############                 UPDATE Q                 ##############
		######################################################################
		# COMPLETE THIS FUNCTION
		# Hints: 
		# - Randomly sample self.n_Dprime data points from the dataset without replacement to make dataset Dprime 
		#   - If the size of the dataset is less than n_Dprime, then sample the whole dataset.
		# - For each datapoint in Dprime
		#   - Train the Q network and update parameters
		#   - Record the loss in self.L_Q and normalize over the batch size
		# - Don't use the 'Softmax' activation here
		T_terminal = 0
		mse_loss = 0
		indices = np.arange(len(D))		# all possible indices
		np.random.shuffle(indices)		# shuffle the indices
		if self.n_Dprime >= len(D):
			# use the whole dataset
			T_terminal = len(D)
			rand_indices = indices[0:T_terminal]
		else:
			T_terminal = self.n_Dprime
			rand_indices = indices[0:T_terminal]

		for t in rand_indices:
			sample 			  = D[t]
			curr_state 		  = sample[0]
			action 			  = sample[1]
			reward	  		  = sample[2]
			next_state 		  = sample[3]

			# get the max Q value achievable
			predicted_label = forwardPass(self.nnQ, curr_state)
			output_label = np.copy(predicted_label)
			output_label[action] = reward + self.gamma * np.max(forwardPass(self.nnQ, next_state))

			mse_loss += computeMSE(predicted_label, output_label)

			# perform backprop
			g = backprop(nn=self.nnQ, X=curr_state, Y=output_label, loss='MSE')

			# update weights for all layers of nnQ
			for j in range(len(self.nnQ)):
				delta_wts  = self.alphaQ * g[j][0] 
				delta_bias = self.alphaQ * g[j][1] 
				self.nnQ[j][0] -= delta_wts / T_terminal
				self.nnQ[j][1] -= delta_bias / T_terminal

		self.L_Q[i] = mse_loss / T_terminal
	

	def update_v(self, tau, T_terminal, i):
		######################################################################
		##############                 UPDATE V                 ##############
		######################################################################
		# COMPLETE THIS FUNCTION
		# Hints: 
		# - For T_terminal steps
		#   - Randomly sample a datapoint from tau
		#   - Train the V network and update parameters
		#   - Record the loss in self.L_V and normalize over the number of steps
		# - Don't use the 'Softmax' activation here
		indices = np.arange(len(tau))
		np.random.shuffle(indices)
		
		mse_loss = 0
		for t in range(T_terminal):
			idx = indices[t]
			sample 			  = tau[idx]
			curr_state 		  = sample[0]
			action 			  = sample[1]
			reward	  		  = sample[2]
			next_state 		  = sample[3]

			predicted_label = forwardPass(self.nnV, X=curr_state, outputLayer=None)
			output_label    = reward + self.gamma * forwardPass(self.nnV, X=next_state, outputLayer=None)

			mse_loss += 0.5 * (predicted_label - output_label) ** 2


			# perform backprop
			g = backprop(nn=self.nnV, X=curr_state, Y=output_label, loss='MSE')

			# update weights for all layers of nnV
			for j in range(len(self.nnV)):
				delta_wts  = self.alphaV * g[j][0] 
				delta_bias = self.alphaV * g[j][1] 
				self.nnV[j][0] -= delta_wts / T_terminal
				self.nnV[j][1] -= delta_bias / T_terminal

		self.L_V[i] = mse_loss / T_terminal
		
	
if __name__ == '__main__':
	A2C = A2C()
	A2C.run_a2c()



    




