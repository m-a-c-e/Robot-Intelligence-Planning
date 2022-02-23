from cmath import sqrt
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
		self.Q = np.zeros([16, 4])	# row idx, col idx, action

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

		self.Q = np.reshape(self.Q, [self.Q.shape[0] * self.Q.shape[1], self.Q.shape[2]])
		self.Q = np.flip(self.Q, axis=1)

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
		s_prev = (0, 0) 	# initial state
		tau = set({})
		self.Q = np.reshape(self.Q, [4, 4, 4])

		for i in range(0, 16):
			# if random value is to be selected
			flag = np.random.choice([True, False], p=[0.2, 0.8])	# 0.2 0.8
			if flag:
				action = np.random.choice([0, 1, 2, 3])		# top, right, down, left
				s_next = self.next_state(s_prev, action)
				r = 0
				if s_next == s_prev:
					r = -2
				else:
					r = self.reward(s_prev)

				tau.add((s_prev, action, s_next, r))
			else:
				value_list = []
				for a in self.A:
					value = self.Q[s_prev[0]][s_prev[1]][a]
					value_list.append(value)
				value_list = np.array(value_list)
				max_value = np.max(value_list)
				idx_list = np.where(value_list == max_value)[0]
				idx = np.random.choice(idx_list)
				action    = idx
				s_next = self.next_state(s_prev, action)
				r = 0
				if s_next == s_prev:
					r = -2
				else:
					r = self.reward(s_prev)
				tau.add((s_prev, action, s_next, r))
			s_prev = s_next

			if r == -1 or r == 1:
				break
		
		for value in tau:
			max_diff = -1
			x_curr = value[0][0]
			y_curr = value[0][1]
			x_next = value[2][0]
			y_next = value[2][1]
			r = value[3]
			action = value[1]
			max_value = np.max(self.Q[x_next][y_next])
			if x_curr == x_next and y_curr == y_next:
				max_diff = max(abs(-2 - self.Q[x_curr][y_curr][action]), max_diff)
				self.Q[x_curr][y_curr][action] = -2 
			elif r != 0:
				max_diff = max(abs(r - self.Q[x_curr][y_curr][action]), max_diff)
				self.Q[x_curr][y_curr][action] = r
			else:
				v1 = r + self.discount * max_value
				max_diff = max(abs(v1 - self.Q[x_curr][y_curr][action]), max_diff)
				self.Q[x_curr][y_curr][action] = r + self.discount * max_value

		if max_diff < self.epsilon:
			return
		

	### HELPER FUNCTION TO USE FOR Q-LEARNING
	# You may choose to write other helper functions or modify these helper functions if you want
	def next_state(self, state, action):
		### IMPLEMENT THIS FUNCTION
		# Given current state and action (and epsilon), return the next state
		next_state = None
		if action == 0:
			next_state = (state[0], state[1] - 1)
		elif action == 1:
			next_state = (state[0] + 1, state[1])
		elif action == 2:
			next_state = (state[0], state[1] + 1)
		else:
			next_state = (state[0] -1, state[1])
		
		if next_state[0] > 3 or next_state[0] < 0:
			return state
		elif next_state[1] > 3 or next_state[1] < 0:
			return state
		else:
			return next_state
	
	def reward(self, state):
		return self.R[state[0]][state[1]]


if __name__ == "__main__":
	p = Problem4()
	policy = p.q_learning()
	print("Policy: " + str(policy))
		