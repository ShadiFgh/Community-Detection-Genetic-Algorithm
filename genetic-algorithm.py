import numpy as np
import random
import networkx as nx

num_parents = 10
num_children = 5
total = num_parents + num_children

with open("sample dataset.txt") as file:
    # number of nodes
    n = int(file.readline())
    file.close()

# loads the given data into an array except the first line
graph = np.loadtxt("sample dataset.txt", skiprows=1, dtype=int)
# number of edges
m =len(graph)

# constructs adjacency matrix for the given graph with n nodes
def adjacency_matrix(graph, n):
    adj = np.zeros((n, n))
    for i in range(0, len(graph)):
        adj[graph[i][0] - 1][graph[i][1] - 1] = 1
        adj[graph[i][1] - 1][graph[i][0] - 1] = 1
    return adj

# computing degrees of nodes
def compute_degree(graph, n):
    adj = adjacency_matrix(graph, n)
    degree = np.zeros((n,1))
    for i in range(0, n):
        degree[i][0] = sum(adj[i])
    return degree

# returns neighbors of each node
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


# creating chromosomes
def create_chromosomes(graph, n, num_parents):
    chromosomes = []
    for i in range(0, num_parents):
        chromosome = []
        for j in range(0, n):
            chromosome.append(random.choice(find_neighbor(graph, n)[j]))
        chromosomes.append(chromosome)
    return chromosomes


# if 2 nodes are in the same community, ci,j = 1, else ci,j =0
def delta(chromosome):
    G = nx.empty_graph()
    for i in range(0, n):
        G.add_edge(i+1, chromosome[i])
    c = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if nx.has_path(G, i+1, j+1):
                c[i][j] = 1
    return c

# print(delta([2, 3, 2, 1, 6, 7, 6, 5]))

def modularity_property(graph, n, chromosomes):
    adj = adjacency_matrix(graph, n)
    degree = compute_degree(graph, n)
    profit = []
    for k in range(0, len(chromosomes)):
        c = delta(chromosomes[k])
        sigma = 0
        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    sigma = sigma + ((adj[i][j] - ((degree[i] * degree[j]) / (2 * m))) * c[i][j])
        profit.append(sigma/(2 * m))
    return np.array(profit)


def choose_parents(profit):
    pi = []
    pn = []
    temp1 = sum(profit)
    for i in range(num_parents):
        temp2 = profit[i] / temp1
        pi.append(temp2)
    pn.append(pi[0])
    for j in range(1, num_parents - 1):
        temp2 = pi[j] + pn[j - 1]
        pn.append(temp2)
    pn.append(1.0)

    tmp1 = np.random.rand()
    p1 = 0;
    while (tmp1 > pn[p1]):
        p1 = p1 + 1
    p1_index = p1
    p2_index = p1_index
    tmp1 = 0
    while (p1_index == p2_index):
        tmp1 = np.random.rand()
        p2 = 0
        while (tmp1 > pn[p2]):
            p2 = p2 + 1
        p2_index = p2
    return p1_index, p2_index

# uniform crossover
def crossover(p):
    p1 = chromosomes[p[0]]
    p2 = chromosomes[p[1]]
    random_vector =[]
    chid = []
    for i in range(0, n):
        random_vector.append(random.choice([0, 1]))
    for i in range(0, n):
        if random_vector[i] == 0:
            chid.append(p1[i])
        else:
            chid.append(p2[i])
    # print(p1)
    # print(p2)
    # print(random_vector)
    return chid


chromosomes = create_chromosomes(graph, n, num_parents)
profit = modularity_property(graph, n, chromosomes)
for i in range(0, num_children):
    p = choose_parents(profit)
    chromosomes.append(crossover(p))
