import networkx as nx
import json
import random
import numpy as np

def invert_labels(labels):
    labs = {}
    for k,v in labels.items():
        if v in labs:
            labs[v] = labs[v] + [k]
        else:
            labs[v] = [k]
    return labs

def data_generator(folder, n):
    values = set()
    i = 0
    while i < n:
        print(i)
        x = random.uniform(0,1)
        if x >0.8:
            y = 4
            g = nx.erdos_renyi_graph(100,0.15)
        elif x <0.8  and x >0.6:
            y = 3
            g = nx.erdos_renyi_graph(100,0.14)
        elif x <0.6 and x>0.4:
            y = 2
            g = nx.erdos_renyi_graph(100,0.12)
        elif x <0.4 and x>0.2:
            y = 1
            g = nx.erdos_renyi_graph(100,0.1)
        else:
            g = nx.erdos_renyi_graph(100,0.08)
            y = 0
        if nx.is_connected(g) == True:
            edges = g.edges()
            values = values.union(set([nx.degree(g,node) for node in g.nodes()]))
            i = i + 1
            edges = [[edge[0],edge[1]]for edge in edges]
            labels = {node:nx.degree(g,node) for node in g.nodes()}
            out = dict()
            out["target"] = y
            out["edges"] = edges
            out["labels"] = labels
            out["inverse_labels"] = invert_labels(labels)
            with open("./"+folder + "/" + str(i) +".json","w") as f:
                json.dump(out,f)
       
data_generator("./erdos_multi_class/train", 500)
data_generator("./erdos_multi_class/test", 500)

