'''
Author: Manan Patel
Colaborator: Shrihari Subramanian
'''

import usa_graph
import copy
import time

##################################################################################
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
no_paths_popped_goal = 0

max_queue_size = -1
max_queue_size_goal = -1

result_path_len = 0
exec_time = 0

depth = -1
goal_reached_flag = False

while(not goal_reached_flag):
    depth = depth + 1
    partial_paths = [[start]]
    visited_list = [start]
    goal_path = []
    max_queue_size_goal = -1
    no_paths_popped_goal = 0

    while(True):
        if partial_paths == []:
            break
        else:
            path = partial_paths[0]    # pick the first path
            if path[-1] == goal:
                # goal reached, break
                goal_path = copy.deepcopy(path)
                goal_reached_flag = True
                break
            elif (len(path) - 1) == depth:
                partial_paths.pop(0)
                no_paths_popped += 1
                no_paths_popped_goal += 1
                continue
            else:

                # pop path from partial_paths
                partial_paths.pop(0)
                no_paths_popped += 1
                no_paths_popped_goal += 1

                # get the neighbors of the head in alphabetical order
                it = G.neighbors(path[-1])
                neighbor_list = []
                for neighbor in it:
                    # continue if neibhor is in visited list
                    if in_list(neighbor, visited_list):
                        continue
                    else:
                        neighbor_list.append(neighbor)
                        update_visited_list(neighbor, visited_list)
                neighbor_list.sort()
                update_visited_list(path[-1], visited_list)

                # create new paths and append to partial paths
                neighbor_list_len = len(neighbor_list)
                for i in range(0, neighbor_list_len):
                    new_path = copy.deepcopy(path) 
                    new_path.append(neighbor_list.pop(-1))
                    partial_paths.insert(0, new_path)
        max_queue_size = max(max_queue_size, len(partial_paths))
        max_queue_size_goal = max(max_queue_size_goal, len(partial_paths))


print(goal_path)
end_time = time.time()

result_path_len = len(goal_path) - 1
exec_time = end_time - start_time

print("Time = ", exec_time)
print("# Paths Popped from Queue (total)= ", no_paths_popped)
print("# Paths Popped from Queue (goal)= ", no_paths_popped_goal)

print("Max Queue Size = (total)", max_queue_size)
print("Max Queue Size = (goal)", max_queue_size_goal)

print("Returned Path's Lenght = ", result_path_len)

