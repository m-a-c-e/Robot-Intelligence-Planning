from pset3 import *

def test_assigned():
    node = [0, 0, 0, -1]
    assert assigned(node) == False, "1. Incorrect function"

    node = [0, 1, 2, 3]
    assert assigned(node) == True, "2. Incorrect function"

def test_constrained_4queens():
    n = 4
    constraints = cmap(n)
    node = [0, 0, 0, 0]
    domains = [[i for i in range(0, n)] for i in range(0, n)]

    assert constrained(node, constraints, domains) == False, "1. Incorrect function"

    node = [1, 3, 0, 2]
    assert constrained(node, constraints, domains) == True, "2. Incorrect function"

def test_constrained_5queens():
    n = 5
    constraints = cmap(n)
    node = [0, 0, 0, 0, 0]
    domains = [[i for i in range(0, n)] for i in range(0, n)]

    assert constrained(node, constraints, domains) == False, "1. Incorrect function"

    node = [0, 3, 1, 4, 2]
    assert constrained(node, constraints, domains) == True, "2. Incorrect function"

def test_constrained_BT():
    for n in range(4, 15):
        nodes, domains, constraints = initialize_parameters(n)
        ans = solve_bt(nodes[0], domains, constraints)
        assert final_constrained(ans, constraints) == True, "1. BT incorrect answer"
   