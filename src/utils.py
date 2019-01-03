import json
import glob
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def read_node_labels(args):
    print("\nCollecting unique node labels.\n")
    labels = set()
    targets = set()
    graphs = glob.glob(args.train_graph_folder + "*.json")
    try:
        graphs = graphs + glob.glob(args.test_graph_folder + "*.json")
    except:
        pass
    for g in tqdm(graphs):
         data = json.load(open(g))
         labels = labels.union(set(list(data["labels"].values())))
         targets = targets.union(set([data["target"]]))
    identifiers = {label:i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print("\n\nThe number of graph classes is: " +str(class_number) + ".\n")
    return identifiers, class_number


def create_logs(args):
    log = dict()
    log["losses"] = []
    log["params"] = vars(args)
    return log

def create_features(data, identifiers):
    """
        
    :param data:
    :param identifiers:
    :return graph:
    :return features:
    """
    graph = nx.from_edgelist(data["edges"])
    features = [[ 1.0 if data["labels"][str(node)] == i else 0.0 for i in range(len(identifiers))] for node in graph.nodes()]
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    return graph, features

def create_batches(graphs, batch_size):
    """
    :param graphs:
    :param batch_size:
    :return batches:
    """
    batches = [graphs[i:i + batch_size] for i in range(0, len(graphs), batch_size)]
    return batches

def calculate_reward(target, predictions):
    reward = (target == torch.argmax(predictions))
    reward = 2*(reward.float()-0.5)
    return reward

def calculate_predictive_loss(data, predictions):
    target = [data["target"]]
    target = torch.tensor(target)
    prediction_loss = torch.nn.functional.nll_loss(predictions, target)
    return target, prediction_loss
