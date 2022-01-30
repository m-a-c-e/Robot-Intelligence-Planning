import numpy as np
import copy
import time
import math

def cmap(n):
	c = np.ndarray((n, n), dtype=dict)
	for i in range(0, n):
		for j in range(0, n):
			dic = {}
			if i == j:
				continue
			for x in range(0, n):
				y_list = []
				for y in range(0, n):
					if x == y:
						continue
					if abs(i - j) == abs(x - y):
						continue
					y_list.append(y)
				dic[x] = y_list
			c[i][j] = dic
	return c


class PSet3():
	def __init__(self):
		self.n 			 = None
		self.domains  	 = None
		self.constraints = None
		self.nodes 		 = None

	def initialize_parameters(self, n):
		self.domains = [[i for i in range(0, n)] for i in range(0, n)]
		self.constraints = cmap(n)
		self.nodes = [[-1] * n] * n

	def standard(self, n=1, test=False):
		if not test:
			n = 1
		time_list = []
		timer_flag = False
		while(not timer_flag):
		# Iterate forever untill time limit is exceeded
			start_time = time.time()
			self.initialize_parameters(n)

			# Standard method
			while(True):
				if time.time() - start_time >= 10:
					timer_flag = True
					break
				if self.nodes == []:
					break
				else:
					# pop the first node
					popped_node = self.nodes.pop(0)

					# check if compelete assignment is there
					if assigned(popped_node):

						# check if satisfies constraints
						if final_constrained(popped_node, self.constraints):
							break
					else:
						# get the index of unassigned node
						idx = popped_node.index(-1)

						# add neighbors
						for i in range(0, n):
							temp_node = copy.deepcopy(popped_node) 
							temp_node[idx] = n - i - 1 
							self.nodes.insert(0, temp_node)

			n += 1
			time_list.append(time.time() - start_time)

			if test:
				return time.time() - start_time

		return time_list
		# return a list of the time taken for each value of n
		# ex: return [1,2,3,5,10]

	def BT(self):
		pasjs
		# return a list of the time taken for each value of n
		# ex: return [1,2,3,5,10]


	def BT_w_FC(self):
		pass
		# return a list of the time taken for each value of n
		# ex: return [1,2,3,5,10]


	def iterative_repair(self):
		pass
		# return a list of the time taken for each value of n
		# ex: return [1,2,3,5,10]


def final_constrained(node, constraints):
	for i in range(0, len(node)):
	# iterate through all the values of the node
		value_i = node[i]
		for j in range(0, len(node)):
		# iterate through all the nodes, except itself
			value_j = node[j]
			if i == j:
				continue

			# get the constraint dictionary
			allowed_values = constraints[i][j].get(value_i)		
			'''
			allowed_values (list) 
				This list represents all the values that node can take at index j
				given that node at index i has the current value
			'''
			try:
				# will throw an error if it doen't exist
				allowed_values.index(value_j)
			except:
				return False
	return True



def assigned(node):
	for i in range(0, len(node)):
		if node[i] == -1:
			return False
	return True


def constrained(node, constraints, domains):
	'''
	node = [0, 0, 2, 3]	
	constraints = [[{}, {}, ...], [], [], ...]
	'''
	domains_cpy = copy.deepcopy(domains)
	for j in range(0, len(node)):
		if not (node[j] in domains_cpy[j]):
			return False
#		if node[j] == -1:
#			break
		for i in range(0, len(node)):
			if j == i:
				continue
			dict_c = constraints[j][i]
			set_c = set(dict_c.get(node[j]))
			d_set = domains_cpy[i]
			inter_set = set_c.intersection(d_set)
			domains_cpy[i] = list(inter_set)

	return True


def constrained_bt(node, domains, constraints):
	domains_cpy = copy.deepcopy(domains)
	for j in range(0, len(node)):
		# iterate through all indices of node / if -1 is encountered end
		if node[j] == -1:
			break
		if not (node[j] in domains_cpy[j]):
			# assignment not possible
			return None
		
		for i in range(0, len(node)):
			# iterate through the constraints between node at j and i
			# skip the indices of constraints where they represent the same variable: (1, 1), (2, 2), ... , (n, n)
			if i == j:
				continue

			dict_c = constraints[j][i]	# get all the constraints for node at j and i
			set_c = set(dict_c.get(node[j]))	# get the available values for node at i based on
												# the assigned value at j (node[j])
			d_set = domains_cpy[i]		# get the set of current available values for node at i
			inter_set = set_c.intersection(d_set)	# check if intersection is possible
			if inter_set == set():
				# inconsistent assignment, break 
				return None
			else:
				# update the domains and continue
				domains_cpy[i] = list(inter_set)
	
	# if all domains are non-empty
	return domains_cpy



def initialize_parameters(n):
	domains = [[i for i in range(0, n)] for i in range(0, n)]
	constraints = cmap(n)
	nodes = [[-1] * n] * n
	
	return nodes, domains, constraints





def solve_n_queens(n):
	nodes, domains, constraints = initialize_parameters(n)
	while(True):
		if nodes == []:
			print("Inconsistent!")
			break
		else:
			# pop the first node
			popped_node = nodes.pop(0)

			# check if compelete assignment is there
			if assigned(popped_node):

				# check if satisfies constraints
				if final_constrained(popped_node, constraints):
					print(popped_node)
					break
			else:
				# get the index of unassigned node
				idx = popped_node.index(-1)

				# add neighbors
				for i in range(0, n):
					temp_node = copy.deepcopy(popped_node) 
					temp_node[idx] = n - i - 1 
					nodes.insert(0, temp_node)			

def revise_BTFC(node, domains, constraints):
	new_domains = copy.deepcopy(domains)
	'''
	for i in range(0, len(node)):
		if node[i] != -1:
			new_domains[i] = [node[i]]
	'''
	
	# need to update domains only for the updated variable
	# get the index of updated node
	# node with -1 as value, minus 1
	idx = node.index(-1) - 1
	new_domains[idx] = [node[idx]]
	value_n = node[idx]

	for i in range(idx + 1, len(node)):
		# update domains as you iterate through nodes which havent
		# been assigned yet 
		curr_set_i = set(new_domains[i])
		valid_set_i = set(constraints[idx][i].get(value_n))
		intersec_set = curr_set_i.intersection(valid_set_i)
		# if not possible, backtrack
		if intersec_set == set():
			return None
		new_domains[i] = list(intersec_set)
	return new_domains


def solve_BTFC(node, domains, constraints):
	idx = -1		
	for i in range(0, len(node)):
		if node[i] == -1:
			idx = i
			break
		
	if idx != -1:

		if domains == None:
			return None

		# assign values from 0 to n - 1 at idx of the node
		# update the domains based on assignment
		# call solve_bt
		for value in range(0, len(node)):
			temp_node = copy.deepcopy(node)
			temp_node[idx] = value
			new_domains = revise_BTFC(temp_node, domains, constraints)
			if new_domains == None:
				continue

			for i in range(0, len(node)):
				# if in the new_domains, only one element is left,
				# assing that value to the node at that index
				if len(new_domains[i]) == 1:
					node[i] = new_domains[i][0]
			
			ans = solve_bt(temp_node, new_domains, constraints)
			if ans != None:
				return ans
	else:
		# value cannot be assigned
		if domains != None:
			# this is the answer
			return (copy.deepcopy(node))
		else:
			# inconsistant final value
			return None


def revise(node, domains, constraints):
	new_domains = copy.deepcopy(domains)
	for i in range(0, len(node)):
		if node[i] != -1:
			new_domains[i] = [node[i]]

	i = 0				# i corresponds to index of domains
	for d_list in new_domains:
		# iterate through domains of all variables
		for j in range(0, len(node)):
			# for each node
			if i == j:
				continue
			set_intrsec = set()
			for d_val in d_list:
				# iterate through values inside domains
				# get the constraints
				set_i = set(constraints[i][j].get(d_val))
				set_j = set(new_domains[j])
				set_x = set_j.intersection(set_i)
				set_intrsec = set_x.union(set_intrsec)
			new_domains[j] = list(set_intrsec)
			if new_domains[j] == []:
				return None
		i += 1
	return new_domains


def revise_bt(node, domains, constraints):
	idx_n = -1
	for i in range(0, len(node)):
		if node[i] == -1:
			idx_n = i - 1
			break
	
	value_n = node[idx_n]
	for i in range(0, idx_n - 1):
	# iterate through all the assignments that have been made
	# check if consistent with it
		value_i = node[i]
		set_i = set(constraints[i][idx_n].get(value_i))
		set_n = set(value_n)
		set_intersec = set_n.intersection(set_i)
		if set_intersec == set():
			return None
	
	domains[idx_n] = list(set_intersec)
	return domains 



def solve_bt(node, domains, constraints):
	# try to assign a value
	idx = -1		
	for i in range(0, len(node)):
		if node[i] == -1:
			idx = i
			break
		
	if idx != -1:

		if domains == None:
			return None

		# assign values from 0 to n - 1 at idx of the node
		# update the domains based on assignment
		# call solve_bt
		for value in range(0, len(node)):
			temp_node = copy.deepcopy(node)
			temp_node[idx] = value
			new_domains = revise(temp_node, domains, constraints)
			if new_domains == None:
				continue

			for i in range(0, len(node)):
				# if in the new_domains, only one element is left,
				# assing that value to the node at that index
				if len(new_domains[i]) == 1:
					node[i] = new_domains[i][0]
			
			ans = solve_bt(temp_node, new_domains, constraints)
			if ans != None:
				return ans
	else:
		# value cannot be assigned
		if domains != None:
			# this is the answer
			return (copy.deepcopy(node))
		else:
			# inconsistant final value
			return None



obj = PSet3()
time_list = obj.standard(3)
print(time_list)