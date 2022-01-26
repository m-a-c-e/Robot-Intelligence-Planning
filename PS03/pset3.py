import numpy as np
import copy

class PSet3():

	def standard(self):
		pass
		# return a list of the time taken for each value of n
		# ex: return [1,2,3,5,10]

	def BT(self):
		pass
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


class Variable():
	def __init__(self, name):
		self.name = name
		self.domain = None
		self.constr = None

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
'''
	for d in domains_cpy:
		if d == set():
			return False
'''

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
				if constrained(popped_node, constraints, domains):
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

solve_n_queens(3)

'''
def solve_bt(n, nodes, domains, constraints):
	while (True):
		if nodes == []:
			return None
		else:
			popped_node = nodes.pop(0)		# pop node from the queue
			new_domains = constrained_bt(popped_node, domains, constraints)
			if new_domains != None:	
			# check if node is constrained
				if assigned(popped_node):
					# if node is completeley assigned, this is the answer
					return popped_node
				else:
					# get the index of unassigned value in the node (represented by -1)
					idx = popped_node.index(-1)

					# add neighbors (make all possible assignments)
					for i in range(0, n):
						temp_node = copy.deepcopy(popped_node) 
						temp_node[idx] = n - i - 1
						nodes.insert(0, temp_node)			
						new_domains[idx] = [n - i - 1]
						return solve_bt(n, nodes, new_domains, constraints)
			else:
				return None
'''

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


n = 15
nodes, domains, constraints = initialize_parameters(n)

ans = solve_bt(nodes[0], domains, constraints)
print(ans)


