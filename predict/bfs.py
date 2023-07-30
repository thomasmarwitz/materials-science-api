from collections import deque
from graph import Graph
from functools import partial


def bfs_to_depth(graph, start_node, max_depth):
    # Initialize all nodes as not visited
    for node in graph:
        graph.nodes[node]["visited"] = False
        graph.nodes[node]["depth"] = float("inf")

    # Create a deque for BFS
    queue = deque([(start_node, 0)])

    # Mark the source node as visited and set its depth as 0
    graph.nodes[start_node]["visited"] = True
    graph.nodes[start_node]["depth"] = 0

    while queue:
        # Dequeue a vertex from queue and print it
        node, depth = queue.popleft()

        # If depth reached is equal to max_depth, stop exploring its neighbors
        if depth == max_depth:
            break

        # Get all neighbors of the dequeued vertex node
        # If a neighbor hasn't been visited, then mark it visited and enqueue it
        for i in graph.neighbors(node):
            if not graph.nodes[i]["visited"]:
                queue.append((i, depth + 1))
                graph.nodes[i]["visited"] = True
                graph.nodes[i]["depth"] = depth + 1

    # Print the nodes and their depths up to max_depth
    return [node for node in graph if graph.nodes[node]["depth"] <= max_depth]


g = Graph.from_path("../data/graph/edges.M.pkl")
g_nx = g.get_nx_graph(2023)

concept = 9390

bfs = partial(bfs_to_depth, graph=g_nx, start_node=concept)
