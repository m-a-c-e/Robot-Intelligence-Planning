from pset4 import *
import math


def test_rectangle():
    e = 0.10
    rect    = Rectangle(0, 0, 10, 10)
    (x, y)  = rect.sample_pt((9, 9), e)

    assert (x <= 10 and x >= 0), "1. Incorrect sample"
    assert (y <= 10 and y >= 0), "2. Incorrect sample"

    # Check if goal is sampled at lest once
    for i in range(0, 100):
        pt = rect.sample_pt((9, 9), e)
        if pt[0] == 9:
            print("Goal sampled")
            break
    
def test_euclidean_dist():
    node1 = Node(1, 1)
    node2 = Node(3, 3)

    dist = round(node1.get_euclidean_dist(node2), 2)
    ans  = round(abs(2 * math.sqrt(2)), 2)

    assert dist == ans, "1. Incorrect distance"

def test_get_nearest():
    # test get nearest
    graph = nx.DiGraph()
    node1 = Node(0, 0)
    node2 = Node(1, 1)
    node3 = Node(2, 2)

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    graph.add_edge(node1, node2)
    graph.add_edge(node1, node3)

    node = Node(3, 3)
    ans = get_nearest(graph, node)

    assert (ans.x  == 2 and ans.y == 2), "1. Incorrect node returned"