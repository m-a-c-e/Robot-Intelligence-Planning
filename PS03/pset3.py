"""
Author: 	 Manan Patel
Colaborator: Shrihari Subramanian
"""


import numpy as np
import copy
import time



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
		self.time_limit  = 600		# seconds
		self.start_time = None
		self.timer_flag = False

	def initialize_parameters(self, n):
		self.domains 	 = [[i for i in range(0, n)] for i in range(0, n)]
		self.constraints = cmap(n)
		self.nodes 		 = [[-1] * n] * n
		self.node 		 = self.nodes[0]

	def standard(self, n=4, test=False):
		if not test:
			n = 4
		time_list = []
		timer_flag = False
		while(not timer_flag):
		# Iterate forever untill time limit is exceeded
			start_time = time.time()
			self.initialize_parameters(n)

			# Standard method
			while(True):
				if time.time() - start_time >= self.time_limit:
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

	def standard_return(self):
		n_list = list(np.arange(4, 12, 1))
		return n_list

	def BT(self, n=4, test=False):
	
		if not test:
			n = 4

		time_list = []
		while not self.timer_flag:
			self.timer_flag = False
			self.initialize_parameters(n)
			self.start_time = time.time()
			ans = solve_bt(self, self.node, self.domains, self.constraints)
			time_list.append(time.time() - self.start_time)
			n += 1
			if test:
				return time_list[0]
		return time_list

	def BT_return(self):
		n_list = list(np.arange(4, 25, 1))
		return n_list

	def BT_w_FC(self, n=4, test=False):
		if not test:
			n = 4
		time_list = []
		while not self.timer_flag:
			self.timer_flag = False
			self.initialize_parameters(n)
			self.start_time = time.time()
			ans = solve_BTFC(self, self.node, self.domains, self.constraints)
			time_list.append(time.time() - self.start_time)
			n += 1
			if test:
				return time_list[0]
		return time_list

	def BT_w_FC_return(self):
		n_list = list(np.arange(4, 31, 1))
		return n_list

	def iterative_repair(self, n=4, test=False):
		if not test:
			n = 4
		time_list = []
		while not self.timer_flag:
			self.timer_falg = False
			self.initialize_parameters(n)
			self.start_time = time.time()
			ans = solve_IR(self, n)
			time_list.append(time.time() - self.start_time)
			n += 1
			if test:
				return time_list[0]
		return time_list

	def iterative_repair_return(self):
		n_list = list(np.arange(4, 16, 1))
		return n_list


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
	#idx = node.index(-1) - 1

	idx = len(node) - 1
	for i in range(0, len(node)):
		if node[i] == -1:
			idx = i - 1
			break

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


def solve_BTFC(obj, node, domains, constraints):
	if time.time() - obj.start_time >= obj.time_limit:
		obj.timer_flag = True
		return None

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
		for value in domains[idx]:
			temp_node 	   = copy.deepcopy(node)
			temp_node[idx] = value
			new_domains    = revise_BTFC(temp_node, domains, constraints)
			if new_domains == None:
				continue

			ans = solve_BTFC(obj, temp_node, new_domains, constraints)
			if ans != None or obj.timer_flag:
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
	idx_n = len(node) - 1
	for i in range(0, len(node)):
		if node[i] == -1:
			idx_n = i - 1
			break
	
	value_n = node[idx_n]
	for i in range(0, idx_n):
	# iterate through all the assignments that have been made
	# check if consistent with it
		value_i = node[i]
		set_i = set(constraints[i][idx_n].get(value_i))
		set_n = set([value_n])
		set_intersec = set_n.intersection(set_i)
		if set_intersec == set():
			return None
	
	domains[idx_n] = [node[idx_n]]
	return domains 


def solve_bt(obj, node, domains, constraints):
	if time.time() - obj.start_time >= obj.time_limit:
		obj.timer_flag = True
		return None
	
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
			new_domains = revise_bt(temp_node, domains, constraints)
			if new_domains == None:
				continue

			ans = solve_bt(obj, temp_node, new_domains, constraints)
			if ans != None or obj.timer_flag:
				return ans
	else:
		# value cannot be assigned
		if domains != None:
			# this is the answer
			return copy.deepcopy(node)
		else:
			# inconsistant final value
			return None


def get_conflicts(v1, v2, i1, i2, constraints):
	valid_values = constraints[i1][i2].get(v1)
	try:
		valid_values.index(v2)
	except:
		return 1
	return 0


def get_total_conflicts(node, idx, constraints):
	"""
	node:	variable assignment
	idx:	variable against which will be checked
	"""
	counter = 0
	for i in range(0, len(node)):
		if i == idx:
			continue
		valid_values = constraints[idx][i].get(node[i])
		try:
			valid_values.index(node[idx])
		except:
			counter += 1
	return counter


def get_node_idx(node, constraints):
	conflict_list = []
	for i in range(0, len(node)):
		conflicts_j = 0
		for j in range(0, len(node)):
			if i == j:
				continue
			# get the valid values between i and j
			valid_values = constraints[i][j].get(node[i])

			try:
				valid_values.index(node[j])		# will throw an error if does not exist in the list
			except:
				conflicts_j += 1
		conflict_list.append(conflicts_j)
	idx = pick_idx(conflict_list)

	return idx


def pick_idx(conflict_list):
	conflict_list = np.array(conflict_list)
	max_conflict = np.amax(conflict_list)
	idx_arr 	 = np.where(conflict_list == max_conflict)[0]					# indices where max conflict
	idx 		 = np.random.choice(idx_arr)

	return idx, max_conflict


def min_conflict_value(node, idx, constraints):
	conflict_list = []	
	for value in range(0, len(node)):
		conflict = 0
		for j in range(0, len(node)):
			if j == idx:
				continue
			conflict += (get_conflicts(value, node[j], idx, j, constraints))
		conflict_list.append(conflict)
	conflict_list = np.array(conflict_list)
	min_value = np.amin(conflict_list)

	return min_value


def new_min_conflict_value(node, idx, constraints):
	conflict_list = []
	for value in range(0, len(node)):
		conflict = 0
		for j in range(0, len(node)):
			if j == idx:
				continue


def solve_IR(node, n, constraints):
	ans = None
	while True:
		if ans == None:
			node = list(np.random.randint(0, n, n, dtype=int))

		# get node index with highest conflict
		idx, num = get_node_idx(node, constraints)
		
		if num == 0:
			# return node
			ans = node
			return ans

		# pick a vlaue for the node at idx which reduces the conflicts to <= num
		min_value = min_conflict_value(node, idx, constraints)	
		if min_value < num and min_value != node[idx]:
			# proceed by updating the value
			node[idx] = min_value
			ans = solve_IR(node, n, constraints)
			return ans
		else:
			node = None


def get_idx_value(node, n, constraints):
	"""
	node:			current variable assignment
	constraints:	dictionaries of constraint relations
	idx: 			represents the variable for which the constraints are being checked 
	"""
	if node == []:
		node = np.arange(0, n, 1)
		np.random.shuffle(node)
		node = list(node)

	# current conflict list
	curr_conf_list = []
	for i in range(0, len(node)):
		value = get_total_conflicts(node, i, constraints)
		curr_conf_list.append(get_total_conflicts(node, i, constraints))
	
	check_list = [0] * len(node)

	if curr_conf_list == check_list:
		return (True, node)
	
	# try possible values for node at index i and see if less conflicts possible
	new_conf_list = []
	new_value_list = []
	for i in range(0, len(node)):
		conf = -1
		new_value = -1
		for value in range(0, len(node)):
			node_cpy 	= copy.deepcopy(node)
			node_cpy[i] = value
			c = get_total_conflicts(node_cpy, i, constraints)
			if conf == -1:
				conf = c
				new_value = value
			elif c < conf:
				conf = c
				new_value = value
		new_conf_list.append(conf)
		new_value_list.append(new_value)
	
	diff_list = np.array(curr_conf_list) - np.array(new_conf_list) 
	max_diff = np.amax(diff_list)
	idx = np.random.choice(np.where(diff_list == max_diff)[0])

	if node[idx] == value or max_diff == 0:
		return False, []
	else:
		node[idx] = value
		return False, node


def solve_IR(obj, n):
	node = []
	stop_flag = False
	while not stop_flag:
		(stop_flag, node) = get_idx_value(node, n, obj.constraints)
		if time.time() - obj.start_time > obj.time_limit:
			obj.timer_flag = True
			return None
	return node



obj = PSet3()
print(obj.standard_return())
print(obj.BT_return())
print(obj.BT_w_FC_return())
print(obj.iterative_repair_return())
#Y1 = obj.standard(11, False)
#X1 = list(np.arange(4, 4 + len(Y1), 1))

# Y2 = obj.BT(30, False)
#X2 = list(np.arange(4, 4 + len(Y2), 1))

#Y3 = obj.BT_w_FC(4, False)
# X3 = list(np.arange(4, 4 + len(Y3), 1))


#Y4 = obj.iterative_repair(4, False)
#X4 = list(np.arange(4, 4 + len(Y4), 1))

# import pandas as pd
#pd.DataFrame(Y2).to_csv('BT.csv')
#print("hello")

