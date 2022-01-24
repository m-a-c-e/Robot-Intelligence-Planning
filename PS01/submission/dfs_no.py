'''
Author: Manan Patel
Colaborator: Srihari Subramanian
'''

import usa_graph
import copy
import time

##################################################################################
def add_to_visited(node_list, list):
    for value in node_list:
        for node in list:
            if node == value:
                break


def update_visited_list(node, list):
    for value in list:
        if value == node:
            return
    list.append(node)


def in_list(node, list):
    for value in list:
        if node == value:
            return True
    return False

##################################################################################
start_time = time.time()
G = usa_graph.load_graph()      # load graph model
start = 'WA'
goal = 'GA'

# Answers
no_paths_popped = 0
max_queue_size = -1
result_path_len = 0
exec_time = 0

partial_paths = [[start]]
goal_path = []

while(True):
    if partial_paths == []:
        break
    else:
        path = partial_paths[0]    # pick the first path
        if path[-1] == goal:
            # goal reached, break
            goal_path = copy.deepcopy(path)
            break
        else:
            # pop path from partial_paths
            partial_paths.pop(0)
            no_paths_popped += 1

            # get the neighbors of the head in alphabetical order
            it = G.neighbors(path[-1])
            neighbor_list = []
            for neighbor in it:
                if path[-1] == neighbor:
                    continue
                else:
                    neighbor_list.append(neighbor)
            neighbor_list.sort()

            # create new paths and append to partial paths
            neighbor_list_len = len(neighbor_list)
            for i in range(0, neighbor_list_len):
                new_path = copy.deepcopy(path) 
                new_path.append(neighbor_list.pop(-1))
                partial_paths.insert(0, new_path)
    max_queue_size = max(max_queue_size, len(partial_paths))

print(goal_path)
end_time = time.time()

result_path_len = len(goal_path) - 1
exec_time = end_time - start_time

print("Time = ", exec_time)
print("# Paths Popped from Queue = ", no_paths_popped)
print("Max Queue Size = ", max_queue_size)
print("Returned Path's Lenght = ", result_path_len)