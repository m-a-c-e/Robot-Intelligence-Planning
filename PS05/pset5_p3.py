import numpy as np


class Problem3():

	def __init__(self):

		# set of states
		self.S = np.asarray([0,1,2,3])

		# set of actions
		self.A = np.asarray([0,1])

		# discount factor or gamma
		self.discount = .95

		# normally V would be initialized randomly, but for consistency - we initialize V to all .2 
		self.V = np.ones(len(self.S))*.2

		# threshold for stopping value iteration
		self.threshold = .05


		### ENTER CODE HERE ###
		# Transition function T(s, a, s')
		# shape (state, action, next_state)
		self.T = np.array([[[.1, .2, .6, .1], [.3, .2, .4, .1]],
              			   [[.3, .2, .3, .2], [.3, .2, .3, .2]],
              			   [[.5, .1, .3, .1], [.2, .1, .6, .1]],
                  		   [[.2, .3, .4, .1], [.3, .3, .3, .1]],
             			  ])

		self.R = np.array([[-.1, -.1], 
						   [-.1, -.1], 
						   [-.1,   1], 
						   [  1, -.1]])
		

	def set_test_vals(self, S, A, T, discount, R, V):
		### DO NOT CHANGE ###
		# This function is for tests on gradescope
		self.S = S
		self.A = A
		self.T = T
		self.discount = discount
		self.R = R
		self.V = V
		

	def value_iteration(self):
		### IMPLEMENT THIS FUNCTION ###
		# returns the final value function as a numpy array and the policy as a numpy array
		# the policy should tell you which action to take for each state
		numStates = np.size(self.S)
		numActions = np.size(self.A)

		policy = None	
		while True:
			value_list = []
			for a in range(0, numActions):
			# for all actions
				z = np.reshape(self.V, [numStates, 1])

				arr = []
				for s in range(0, numStates):
					x = self.T[s, a, :]
					y = self.R[s, a]
					ans = np.matmul(x, (y + self.discount * z))[0]
					arr.append(ans)
				value_list.append(arr)

			value_list = np.array(value_list).transpose()
			# get the max values and their indiceso
			max_list = np.max(value_list, axis=1)
			policy = np.argmax(value_list, axis=1)

			# policy	
			diff_list = abs(np.reshape(max_list, numStates) - self.V)

			max_diff = np.max(diff_list)
			self.V = np.reshape(max_list, numStates)	

			if max_diff < self.threshold:
				break

		return self.V, policy 

if __name__ == "__main__":

	p = Problem3()
	V, policy = p.value_iteration()
	print("Value Function: " + str(V))
	print("Policy: " + str(policy))

