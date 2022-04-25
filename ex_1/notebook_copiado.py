
Structure of networks: Degree and transitivity

Francisco Aparecido Rodrigues, francisco@icmc.usp.br.
Universidade de São Paulo, São Carlos, Brasil.
https://sites.icmc.usp.br/francisco
Copyright: Creative Commons

In this lecture, we will learn about connectivity and transitivity in networks.

We need some library to process the data. Let us import the Numpy library because we will work with vectors.

from numpy  import *
import numpy as np

We have to import the matplotlib library to plot the results.

import matplotlib.pyplot as plt

To process the networks, we consider the Networkx library. This library can be installed following the steps in this link: https://networkx.github.io/documentation/latest/install.html

import networkx as nx

To read the network from a file, we use the command read_edgelist. Notice that the network format is the edge list and we consider the weighted version, because the file can have three columns (two for the connection and the third one for the weigth of the connections). In our case, we ignore the third connection, because we are considering only unweighted networks.

G= nx.read_edgelist("data/lesmis.txt", nodetype=int, data=(('weight',float),))
# If the data file has only two columns, use this:
#G= G=nx.read_edgelist("data/powergrid.txt", nodetype=int)

pos = nx.spring_layout(G)
nx.draw(G, pos, node_color="b", node_size=50, with_labels=False)

We transfor the network into the undirected version. In our case, this step is not be necessary because the default output from read_edgelist is an undirected graph. To obtain a directed graph, we need to include the field "create_using= nx.DiGraph()" as input. Anyway, let us perform this transformation here to exemplify this operation.

G = G.to_undirected()
G.remove_edges_from(nx.selfloop_edges(G))

We consider only the largest component here. The selection of the largest component is important mainly in the calculation of the measures related to distance, because the distance between pairs of nodes is defined only for nodes in the same component. Remember that a component is a subgraph in which all of its nodes can access each other.

Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G = G.subgraph(Gcc[0])

Sometimes the node labels are not in the sequential order or labels are used. To facilitate our implementation, let us convert the labels to integers starting with the index zero, because Python uses 0-based indexing.

G = nx.convert_node_labels_to_integers(G, first_label=0)

Let us verify the number of nodes and edges of the network.

N = len(G)
M = G.number_of_edges()
print('Number of nodes:', N)
print('Number of edges:', M)

Number of nodes: 77
Number of edges: 254

Now, let us calculate measures to characterize the network structure.
Degree related measures

Since the degree() function from Networkx provides an iterator for (node, degree) , we will use only the values of the degree and ignore the label of the nodes. We also convert the list to a numpy array, since this structure is easier to be manipuled than other structures.

vk = dict(G.degree())
vk = list(vk.values())
vk = np.array(vk)
print('Degree:', vk)

Degree: [ 1 10  3  3  1  1  1  1  1  1 36  1  2  1  1  1 15 11 16 11 17  4  8  4
  1  2  6  6  6  6  6  3  2 22  7  7 19 15 13 10 10 10  9  3  7  9  7  7
  7  7  7  2 11  3  2  3  1  7  4  1  2 11 13  2  1 11  9 11 12 12 10  2
  2  7  2  1  1]

The mean degree of the network:

md = mean(vk)
print('Mean degree: ', md)

Mean degree:  6.597402597402597

From the node degrees, we can calculate several statistical measures. Let us develop a function to calculate the degree distribution.

def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values())  # we get only the degree values
    vk = np.array(vk)
    maxk = np.max(vk)
    mink = np.min(vk)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

Thus, to obtain the degree distribution, we have:

ks, Pk = degree_distribution(G)

Thus, we can plot the degree distribution in log-log scale.

fig = plt.subplot(1,1,1)
fig.set_xscale('log')
fig.set_yscale('log')
plt.plot(ks,Pk,'bo')
plt.xlabel("k", fontsize=20)
plt.ylabel("P(k)", fontsize=20)
plt.title("Degree distribution", fontsize=20)
#plt.grid(True)
plt.savefig('degree_dist.eps') #save the figure into a file
plt.show(True)

We can also calculate the statistical moments of the degree distribution. A function to calculate the m-th moment of the degree distribution is defined as:

def momment_of_degree_distribution(G,m):
    k,Pk = degree_distribution(G)
    M = sum((k**m)*Pk)
    return M

Or, we can also calculate through:

def momment_of_degree_distribution2(G,m):
    M = 0
    N = len(G)
    for i in G.nodes:
        M = M + G.degree(i)**m
    M = M/N
    return M

The first statistical moment is equal to the mean degree:

k1 = momment_of_degree_distribution(G,1)
print("Mean degree = ", mean(vk))
print("First moment of the degree distribution = ", k1)

Mean degree =  6.597402597402597
First moment of the degree distribution =  6.5974025974025965

And the second momment of the degree distribution:

k2 = momment_of_degree_distribution(G,2)
print("Second moment of the degree distribution = ", k2)

Second moment of the degree distribution =  79.53246753246754

The variance is calculated as: V(k)=⟨k2⟩−k2

variance = momment_of_degree_distribution(G,2) - momment_of_degree_distribution(G,1)**2
print("Variance of the degree = ", variance)

Variance of the degree =  36.00674650025301

The level of network heterogeneity with respect to the number of connections can be quantified by the Shannon entropy. A function to calculate the Shannon entropy of the degree distribution can be defined as:

def shannon_entropy(G):
    k,Pk = degree_distribution(G)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

The Shannon entropy of P(k)

H = shannon_entropy(G)
print("Shannon Entropy = ", "%3.4f"%H)

Shannon Entropy =  3.5957

We can also calculate the normalize version of the entropy (Hn=H/Hmax
, where Hmax=logN

)

def normalized_shannon_entropy(G):
    k,Pk = degree_distribution(G)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H/math.log(len(G),2)

H = normalized_shannon_entropy(G)
print("Normalized Shannon Entropy = ", "%3.4f"%H)

Normalized Shannon Entropy =  0.5738

Transitivity and clustering

In addition to the degree, another important property of networks is related to the number of triangles, which is related to the concept of transitivity. The transitivity of the network G is calculated as:

CC = (nx.transitivity(G)) 
print("Transitivity = ","%3.4f"%CC)

Transitivity =  0.4989

The level of triangles in a network can also be quantified by the average clustering coefficient, calculated from the local clustering coefficient, i.e.,

avc = nx.average_clustering(G)
print("Average clustering:", "%3.4f"%avc)

Average clustering: 0.5731

The clustering of each node is defined by the fraction of edges among the neighbors of each node.

vcc = []
for i in G.nodes():
    vcc.append(nx.clustering(G, i))
vcc= np.array(vcc)
print('Clustering of all nodes:', vcc)

Clustering of all nodes: [0.         0.06666667 1.         1.         0.         0.
 0.         0.         0.         0.         0.12063492 0.
 1.         0.         0.         0.         0.31428571 0.49090909
 0.40833333 0.38181818 0.32352941 0.33333333 0.64285714 0.66666667
 0.         1.         1.         1.         1.         1.
 1.         1.         1.         0.35497835 0.47619048 0.42857143
 0.33333333 0.60952381 0.76923077 0.8        0.8        0.71111111
 0.83333333 1.         1.         0.61111111 1.         1.
 1.         1.         1.         1.         0.45454545 1.
 0.         0.33333333 0.         0.9047619  1.         0.
 0.         0.69090909 0.75641026 0.         0.         0.92727273
 1.         0.92727273 0.86363636 0.86363636 0.93333333 1.
 1.         1.         1.         0.         0.        ]

The statistical distribution of the clustering coefficient:

plt.figure()
plt.hist(vcc, bins  = 10, density=True)
plt.title("Distribution of the clustering coefficient", fontsize=20)
plt.ylabel("P(cc)", fontsize=15)
plt.xlabel("Clustering coefficient (cc)", fontsize=15)
#plt.grid(True)
plt.savefig('clustering.eps') #save the figure into a file
plt.show()

The clustering in function of the degree can reveal a hierarchical organization if C(k)≈k−β

.

#Average clustering for each degree k
ck = list()
ks = list()
for k in np.arange(np.min(vk), np.max(vk)):
    aux = vk == k
    if(len(vcc[aux]) > 0):
        cm = mean(vcc[aux]) #average clustering among all the nodes with degree k
        ck.append(cm)
        ks.append(k)
plt.loglog(ks,ck,'bo')
plt.title("Clustering coefficient according to degree", fontsize=20)
plt.ylabel("cc(k)", fontsize=15)
plt.xlabel("k", fontsize=15)
#plt.grid(True)
plt.savefig('cck.eps')
plt.show(True)

Exercises

1 - For the lesmis dataset, calculate the third and fourth statistical moments.

2 - Define a function to create the complexity measure and calculate for several networks in the data set. This coefficient is defined as: Cx=E[K2]E[K]

3 - Calculate the degree distribution of the power-grid network. Also, obtain the complexity index.

4 - Construct a function that returns the following networks properties:

    Number of nodes
    Number of connections
    Averge degree
    Second moment of degree distribution
    Average clustering coefficient
    Transitivity
    Shanon entropy of the degree distribution

    Consider the following networks:
    Power grid
    US airport
    Euro road
    Protein interactions (ppi.maayan-vidal.txt)

5 - Construct a function that returns the degree distribution in log or linear scale. The function can construct scatterplot or a bar plot according to its parameters.

 

