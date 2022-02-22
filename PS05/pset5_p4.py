import numpy as np
import copy

class Problem4():

	def __init__(self):
		# set of states
		# Format: 4x4 grid of coordinates
		self.S = []
		for i in range(4):
			for j in range(4):
				self.S.append((i,j))

		# set of actions (up, right, down, left)
		self.A = np.asarray([0,1,2,3])

		# discount factor or gamma
		self.discount = .95

		# reward for each state
		self.R = [[0, 0, 0, 0], [0, 0, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]]

		# start state (top right corner)
		self.start_state = (0, 0)

		# actions are taken with probability .8
		# a random action is taken with probability .2
		self.epsilon = .2


		### ENTER CODE HERE ###
		# Create the Q table
		raise NotImplementedError("q_learning init function not complete")
		self.Q = None

	def set_test_vals(self, S, A, discount, R):
		### DO NOT CHANGE ###
		# This function is for tests on gradescope
		self.S = S
		self.A = A
		self.discount = discount
		self.R = R


	def q_learning(self):
		### DO NOT CHANGE ###
		# returns the final policy as a numpy array
		# the policy should be the Q table for each state-action value

		for i in range(100000):
			self.rollout()

		return self.Q

	def rollout(self):
		### IMPLEMENT THIS FUNCTION

		# This function rolls out an iteration of Q-learning to update the Q table.
		# Hints: 
		# - Not all actions are possible in each spot.  For example, if you are at the top of the grid, you can't go up
		# 	- If a move isn't possible - the Q-value should be set to -2
		# - Start each iteration at the start location
		# - The reward locations on the map are static
		# - An action should be taken with epsilon-greedy probability self.epsilon
		# 	- This means that the action taken is not deterministic
		# 	- You may account for this in the rollout or the next_state function

		raise NotImplementedError("q_learning rollout function not complete")


	### HELPER FUNCTION TO USE FOR Q-LEARNING
	# You may choose to write other helper functions or modify these helper functions if you want
	def next_state(self, state, action):
		### IMPLEMENT THIS FUNCTION
		# Given current state and action (and epsilon), return the next state
		pass



if __name__ == "__main__":
	p = Problem4()
	policy = p.q_learning()
	print("Policy: " + str(policy))
		