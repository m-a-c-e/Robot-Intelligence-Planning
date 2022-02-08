import networkx as nx
import math
import numpy as np
import pddlpy
import random

import matplotlib.pyplot as plt

from shapely.geometry import LineString
from dfs_yes import *

from planning_graph.planning_graph import PlanningGraph
from planning_graph.planning_graph_planner import GraphPlanner


class Node:
    """
    Implements a data structure to store point locations.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get_loc(self):
        """
        Returns the co-ordinates of the node
        """
        return (self.x, self.y)

    def get_euclidean_dist(self, node):
        """
        Returns the euclideans distance with respect to node
        """
        return abs(math.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2))


class Rectangle:
    """
    Implements a data structure to represent rectangular objs.
    """
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def sample_pt(self, goal_loc, e):
        """
        Generates a sample point within the bounds of the rectangle
        inclusive of edges
        Args:
            goal_loc (tuple)    : (x, y) co-ordinate
            e                   : probability with which goal must be sampled

        Returns:
            tuple of x and y location
        """
        prob_list = [1 - e, e]            # Sample the goal with 0.10 probability
        sample_goal = [True, False]

        while (True):
            if np.random.choice(sample_goal, size=1, replace=True, p=prob_list)[0]:
                x = random.uniform(self.x1, self.x2)
                y = random.uniform(self.y1, self.y2)
                if x == self.x1 or y == self.y1:
                    continue
                return (round(x, 3) , round(y, 3))
            else:
                x = goal_loc[0]
                y = goal_loc[1]
                return (x, y)


def get_nearest(graph, rand_node):
    """
    Gets the node nearest to rand_node in the graph.
    Euclidean distance is used.

    Args:
    graph (networkx obj): represents the graph
    rand_node (Node)    : random node

    Returns:
    nearest_node (Node): Node nearest to rand_node
    """
    nearest_node = None

    # get all the nodes
    nodes_list = list(graph.nodes())
    min_dist = float('inf')
    for node in nodes_list:
        dist = rand_node.get_euclidean_dist(node)
        if min_dist > dist:
            min_dist     = dist
            nearest_node = node

    return nearest_node


def path_exists(nearest_node, rand_node, corners):
    """
    Checks if a valid line can be drawn between nearest_node
    and rand_node without intersecting the obstacle

    Args:
    graph (networkx obj): represents the graph
    rand_node (Node)    : random node
    nearest_node(Node)  : Node nearest to rand_node
    objstacle (Rectangle): represents the obstacle in the room

    Returns:
    True    : if a valid path exists (line can be drawn without hitting the obstacle)
    Flase   : if a valid path does not exist
    """
    # get the line joining nearest_node and rand_node
    pt1         = (nearest_node.x, nearest_node.y)
    pt2         = (rand_node.x, rand_node.y)
    pt_line     = LineString([pt1, pt2])

    if len(corners) < 4:
        return True

    for i in range(0, len(corners)):
        corner1 = corners[i]
        if i == len(corners) - 1:
            corner2 = corners[0]
            rect_line = LineString([corner1, corner2])
            if pt_line.intersects(rect_line):
                return False        # path doesnt exist
        else:
            corner2 = corners[i + 1]
            rect_line = LineString([corner1, corner2])
            if pt_line.intersects(rect_line):
                return False        # path doesnt exist

    return True


class PSet4():

    def solve_pddl(self, domain_file: str, problem_file: str):
        """

        domain_file: str - the path to the domain PDDL file
        problem_file: str - the path to the problem PDDL file

        returns: a list of Action as Strings , or None if problem is infeasible
        """
        domprob = pddlpy.DomainProblem('domain.pddl', 'problem.pddl')
        print(domprob.initialstate())
        print(domprob.operators())


        planning_graph = PlanningGraph('domain.pddl', 'problem.pddl')
        graph          = planning_graph.create()
        goal           = planning_graph.goal
        graph_planner  = GraphPlanner()
        plan = graph_planner.plan(graph, goal)

        final_plan = []
        for plan_obj in plan.data.values():
            for x in plan_obj._plan:
                op  = x.operator_name
                if op == 'NoOp':
                    continue
                '''
                prop= x.precondition_pos
                var = x.variable_list
                '''
                final_plan.append(op)

        return final_plan

    def solve_rrt(self, corners):
        """
        corners: [(float, float)] - a list of 4 (x, y) corners in a rectangle, in the
           order upper-left, upper-right, lower-right, lower-left

        returns: a list of (float_float) tuples containing the (x, y) positions of
           vertices along the path from the start to the goal node. The 0th index
           should be the start node, the last item should be the goal node. If no
           path could be found, return None
        """
        plt.plot([1], [1], label='S', marker='X', markersize=10, color='g')
        plt.plot(9, 9, label='G', marker='X', markersize=10, color='g')

        for i in range(0, len(corners)):
            corner1 = corners[i]
            if i == len(corners) - 1:
                corner2 = corners[0]
                plt.plot(corner1, corner2, color='r')
            else:
                corner2 = corners[i + 1]
                plt.plot(corner1, corner2, color='r')
        
        # initialize parameters
        start_loc  = (1, 1)      
        start_node = Node(start_loc[0], start_loc[1])

        goal_loc   = (9, 9)
        goal_node  = Node(goal_loc[0], goal_loc[1])

        iterations = 1000
        room       = Rectangle(0, 0, 10, 10)
        e          = 0.10                       # probability with which goal will be sampled

        # initialize graph with start node
        graph = nx.DiGraph()
        graph = graph.to_undirected()
        graph.add_node(start_node)

        for i in range(0, iterations):
            # sample a point within the room dimensions
            (x_rand, y_rand)    = room.sample_pt(goal_loc, e)
            rand_node           = Node(x_rand, y_rand)

            # get the nearest node to this node
            nearest_node        = get_nearest(graph, rand_node)

            # check if path can be draw between these two nodes
            if path_exists(nearest_node, rand_node, corners):
                graph.add_edge(nearest_node, rand_node)
                plt.plot([nearest_node.x, rand_node.x], [nearest_node.y, rand_node.y], color='b')

#        uncomment to see the graph
#        plt.show()
#        plt.autoscale()

        # graph is made. Now implement depth first search to find a goal
        path = dfs(graph, start_node, goal_node)

        ans = []
        for node in path:
            ans.append((node.x, node.y))
        return ans


if __name__ == "__main__":
    p = PSet4()
    plan = p.solve_pddl('domain.pddl', 'problem.pddl')
    print("Plan : ", plan)
    print("Plan Length : ", len(plan))

    rrt_path = p.solve_rrt([()])
    print("Path: ", rrt_path)
    print("Path length: ", len(rrt_path))

    '''
    corners  = [(3, 7), (7, 7), (7, 3), (3, 3)]
    rrt_path = p.solve_rrt(corners)
    print("Path: ", rrt_path)
    print("Path length: ", len(rrt_path))
    '''






