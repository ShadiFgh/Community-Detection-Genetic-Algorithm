import numpy as np


with open("sample dataset.txt") as file:
    #number of nodes
    n = int(file.readline())
    file.close()

#loads the given data into an array ecept the first line
graph = np.loadtxt("sample dataset.txt", skiprows=1, dtype=int)

# construts adjacency matrix for the given graph with n nodes
def adjacency_matrix(graph, n):
    adj = np.zeros((n, n))
    for i in range(0, len(graph)):
        adj[graph[i][0] - 1][graph[i][1] - 1] = 1
        adj[graph[i][1] - 1][graph[i][0] - 1] = 1
    return adj

