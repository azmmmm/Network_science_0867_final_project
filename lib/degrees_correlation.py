import networkx as nx
import seaborn as sn
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def self_degree(graph,node,t=''):
    if(type(graph)==nx.classes.graph.Graph):
        return graph.degree(node)
    elif(t=='out'):
        return graph.out_degree(node)
    elif(t=='in'):
        return graph.in_degree(node)

def mean_s(li):
    if(len(li)==0):
        return 0
    else:
        return mean(li)

def degrees_correlation_analise(graph,weight=None,x='out', y='in'):
    plt.figure(1)

    mix_mat=nx.degree_mixing_matrix(graph,x,y)
    sn.heatmap(mix_mat,xticklabels='', yticklabels='',cmap='Blues')
    print(nx.degree_pearson_correlation_coefficient(graph,weight=weight,x='out', y='in'))
    knn={}
    for node in graph.nodes():
        neig_degree=[]
        for neig in graph.neighbors(node):
            neig_degree.append(self_degree(graph,neig,t=y))
        source_degree=self_degree(graph,node,x)
        if(knn.get(source_degree)):
            knn[source_degree].append(mean_s(neig_degree))
        else:
            knn[source_degree]=[mean_s(neig_degree)]
    x=[]
    y=[]
    for degree in knn:
        x.append(degree)
        y.append(mean_s(knn[degree]))
    plt.figure(2)
    plt.xlabel("k")
    plt.ylabel("K_annd(k)")
    plt.scatter(x, y)
    plt.show()
    
