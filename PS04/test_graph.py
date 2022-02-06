import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G = G.to_undirected()

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_loc(self):
        return (self.x, self.y)

node = Node(1, 1)
G.add_node(node)


node1 = Node(1, 2)
G.add_node(node1)
G.add_edge(node, node1)
a = list(G.neighbors(node1))
print(a)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=100)
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
nx.draw_networkx_labels(G, pos, font_size=5)
plt.show()
