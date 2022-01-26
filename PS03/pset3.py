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