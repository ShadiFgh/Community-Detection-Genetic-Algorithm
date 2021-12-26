import numpy as np
import random
import networkx as nx

num_parents = 10
num_chidren = 5

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

#computing degrees of nodes
def compute_degree(graph, n):
    adj = adjacency_matrix(graph, n)
    degree = np.zeros((n,1))
    for i in range(0, n):
        degree[i][0] = sum(adj[i])
    return degree

#returns neighbors of each node
def find_neighbor(graph, n):
    adj = adjacency_matrix(graph,n)
    neighbors = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if adj[i][j] == 1:
                row.append(j+1)
        neighbors.append(row)
    return neighbors


#creating chromosomes
def create_chromosomes(graph, n, num_parents):
    chromosomes = []
    for i in range(0, num_parents):
        chromosome = []
        for j in range(0, n):
            chromosome.append(random.choice(find_neighbor(graph, n)[j]))
        chromosomes.append(chromosome)
    return chromosomes


#if 2 nodes are in the same community, ci,j = 1, else ci,j =0
def delta(chromosome):
    n = len(chromosome)
    G = nx.empty_graph()
    for i in range(0, len(chromosome)):
        G.add_edge(i+1, chromosome[i])
    c = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if nx.has_path(G, i+1, j+1):
                c[i][j] = 1
    return c

# print(delta(create_chromosomes(graph, n, num_parents)[0]))