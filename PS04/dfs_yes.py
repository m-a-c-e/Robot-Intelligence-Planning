'''
Author: Manan Patel
Colaborator: Srihari Subramanian

'''

import copy

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
def dfs(graph, start, goal):
    G = graph      # load graph model

    partial_paths = [[start]]
    visited_list = [start]
    goal_path = []

    while(True):
        if partial_paths == []:
            break
        else:
            path = partial_paths[0]    # pick the first path
            if path[-1].x == goal.x and path[-1].y == goal.y:
                # goal reached, break
                goal_path = copy.deepcopy(path)
                break
            else:
                # get the neighbors of the head in alphabetical order
                it = G.neighbors(path[-1])

                # pop path from partial_paths
                partial_paths.pop(0)

                neighbor_list = []
                for neighbor in it:
                    # continue if neibhor is in visited list
                    if in_list(neighbor, visited_list):
                        continue
                    else:
                        neighbor_list.append(neighbor)
                        update_visited_list(neighbor, visited_list)

                # create new paths and append to partial paths
                neighbor_list_len = len(neighbor_list)
                for i in range(0, neighbor_list_len):
                    new_path = copy.deepcopy(path) 
                    new_path.append(neighbor_list.pop(-1))
                    partial_paths.insert(0, new_path)

    return goal_path
