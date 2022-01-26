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

def test_solve_4_queens():

    for i in range(4, 9):
        solve_n_queens(i)

def test_constrained_6queens():
    n = 6
    nodes, domains, constraints = initialize_parameters(n)
    ans = [0, 2, 4, 6, 1, 3, 5]

    assert constrained(ans, domains, constraints) == True
    