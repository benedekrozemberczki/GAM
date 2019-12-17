"""Data reading utils."""

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
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def read_node_labels(args):
    """
    Reading the graphs from disk.
    :param args: Arguments object.
    :return identifiers: Hash table of unique node labels in the dataset.
    :return class_number: Number of unique graph classes in the dataset.
    """
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
    identifiers = {label: i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print("\n\nThe number of graph classes is: "+str(class_number)+".\n")
    return identifiers, class_number

def create_logs(args):
    """
    Creates a dictionary for logging.
    :param args: Arguments object.
    :param log: Hash table for logs.
    """
    log = dict()
    log["losses"] = []
    log["params"] = vars(args)
    return log

def create_features(data, identifiers):
    """
     Creates a tensor of node features.
    :param data: Hash table with data.
    :param identifiers: Node labels mapping.
    :return graph: NetworkX object.
    :return features: Feature Tensor (PyTorch).
    """
    graph = nx.from_edgelist(data["edges"])
    features = []
    for node in graph.nodes():
        features.append([1.0 if data["labels"][str(node)] == i else 0.0 for i in range(len(identifiers))])
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    return graph, features

def create_batches(graphs, batch_size):
    """
    Creating batches of graph locations.
    :param graphs: List of training graphs.
    :param batch_size: Size of batches.
    :return batches: List of lists with paths to graphs.
    """
    batches = [graphs[i:i + batch_size] for i in range(0, len(graphs), batch_size)]
    return batches

def calculate_reward(target, prediction):
    """
    Calculating a reward for a prediction.
    :param target: True graph label.
    :param prediction: Predicted graph label.
    """
    reward = (target == torch.argmax(prediction))
    reward = 2*(reward.float()-0.5)
    return reward

def calculate_predictive_loss(data, predictions):
    """
    Prediction loss calculation.
    :param data: Hash with label.
    :param prediction: Predicted label.
    :return target: Target tensor.
    :prediction loss: Loss on sample.
    """
    target = [data["target"]]
    target = torch.tensor(target)
    prediction_loss = torch.nn.functional.nll_loss(predictions, target)
    return target, prediction_loss
