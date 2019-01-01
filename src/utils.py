import json
import glob
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
    print("Collecting unique node labels.\n")
    labels = set()
    targets = set()
    graphs = glob.glob(args.train_graph_path + "*.json")
    try:
        graphs = graphs + glob.glob(args.test_graph_path + "*.json")
    except:
        pass
    for g in tqdm(graphs):
         data = json.load(open(g))
         labels = labels.union(set(list(data["labels"].values())))
         targets = targets.union(set([data["target"]]))
    identifiers = {label:i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print("The number of graph classes is: " +str(class_number) + ".\n")
    return identifiers, class_number

