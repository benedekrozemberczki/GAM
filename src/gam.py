import glob
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import networkx as nx
import torch.nn.functional as F
from utils import read_node_labels, create_logs

class StepNetworkLayer(torch.nn.Module):
     """
     Step Network Layer Class for selecting next node to move.
     """
     def __init__(self, args, identifiers):
         """
         Initializing the layer.
         :param args: Arguments object.
         :param identifiers: Node type -- id hash map.
         """
         super(StepNetworkLayer, self).__init__()
         self.identifiers = identifiers
         self.args = args
         self.setup_attention()
         self.create_parameters()

     def setup_attention(self):
         """
         Initial attention generation with uniform attention scores.
         """
         self.attention = torch.ones((len(self.identifiers)))/len(self.identifiers)

     def create_parameters(self):
         """
         Creating trainable weights and initlaizing them.
         """
         self.theta_step_1 = torch.nn.Parameter(torch.Tensor(len(self.identifiers), self.args.step_dimensions))
         self.theta_step_2 = torch.nn.Parameter(torch.Tensor(len(self.identifiers), self.args.step_dimensions))
         self.theta_step_3 = torch.nn.Parameter(torch.Tensor(2*self.args.step_dimensions, self.args.combined_dimensions))
         torch.nn.init.uniform_(self.theta_step_1,-1,1)
         torch.nn.init.uniform_(self.theta_step_2,-1,1)
         torch.nn.init.uniform_(self.theta_step_3,-1,1)

     def sample_node_label(self, original_neighbors, graph, features):
         """
         Sampling a label from the neighbourhood.
         :param original_neighbors: Neighbours of the source node.
         :param graph: NetworkX graph.
         :param features: Node feature matrix.
         :return label: Label sampled from the neighbourhood with attention.
         """
         neighbor_vector = torch.tensor([1.0 if node in original_neighbors else 0.0 for node in graph.nodes()])
         neighbor_features = torch.mm(neighbor_vector.view(1,-1), features)
         attention_spread = self.attention * neighbor_features
         normalized_attention_spread = attention_spread / attention_spread.sum()
         normalized_attention_spread = normalized_attention_spread.detach().numpy().reshape(-1)
         label = np.random.choice(np.arange(len(self.identifiers)), p=normalized_attention_spread)
         return label

     def make_step(self, node, graph, features, labels, inverse_labels):
         """
         :param node: Source node for step.
         :param graph: NetworkX graph.
         :param features: Feature matrix.
         :param labels: Node labels hash table. 
         :param inverse_labels: Inverse node label hash table.
         """
         original_neighbors = set(nx.neighbors(graph, node))
         label = self.sample_node_label(original_neighbors, graph, features)
         new_node = random.choice(list(set(original_neighbors).intersection(set(inverse_labels[str(label)]))))
         new_node_attributes = torch.zeros((len(self.identifiers),1))
         new_node_attributes[labels[str(label)],0] = 1.0
         attention_score = self.attention[labels[str(label)]]
         return new_node_attributes, new_node, attention_score
         
     def forward(self, data, graph, features,node):
         """
         Making a forward propagation step.
         :param data: Data hash table.
         :param graph: NetworkX graph object.
         :param features: Feature matrix of the graph.
         :param node: Base node where the step is taken from.
         :return state: State vector.
         :return node: New node to move to.
         :return attention_score: Attention score of chosen node. 
         """
         feature_row, node, attention_score = self.make_step(node, graph, features, data["labels"], data["inverse_labels"])
         hidden_attention = torch.mm(self.attention.view(1, -1), self.theta_step_1)
         hidden_node = torch.mm(torch.t(feature_row), self.theta_step_2)
         combined_hidden_representation = torch.cat((hidden_attention, hidden_node), dim = 1)
         state = torch.mm(combined_hidden_representation, self.theta_step_3).view(1, 1, self.args.combined_dimensions)
         return state, node, attention_score
         
class DownStreamNetworkLayer(torch.nn.Module):      
    """
    Neural network layer for attention update and node label assignment.
    """
    def __init__(self, args, target_number, identifiers):
        """
        
        :param args:
        :param target_number:
        :param identifiers:
        """
        super(DownStreamNetworkLayer, self).__init__()
        self.args = args 
        self.target_number = target_number
        self.identifiers = identifiers
        self.create_parameters()

    def create_parameters(self):
         """
         Defining and initializing the classification and attention update weights.
         """
         self.theta_classification = torch.nn.Parameter(torch.Tensor(self.args.combined_dimensions, self.target_number))
         self.theta_rank = torch.nn.Parameter(torch.Tensor(self.args.combined_dimensions, len(self.identifiers)))
         torch.nn.init.xavier_normal_(self.theta_classification)
         torch.nn.init.xavier_normal_(self.theta_rank)

    def forward(self, hidden_state):
        """
        Making a forward propagation pass with the input from the LSTM layer.
        :param hidden_state: LSTM state used for labeling and attention update.
        """
        predictions = torch.mm(hidden_state.view(1,-1), self.theta_classification)
        attention = F.softmax(torch.mm(hidden_state.view(1,-1), self.theta_rank), dim = 1)
        return predictions, attention
 
class GAM(torch.nn.Module):

    def __init__(self,args):
        super(GAM, self).__init__()
        self.args = args
        self.identifiers, self.class_number = read_node_labels(self.args)
        self.step_block = StepNetworkLayer(self.args, self.identifiers)
        self.recurrent_block = torch.nn.LSTM(self.args.combined_dimensions, self.args.combined_dimensions, 1)
        self.down_block = DownStreamNetworkLayer(self.args, self.class_number, self.identifiers)
        self.reset_attention()

    def reset_attention(self):
        self.step_block.attention = torch.ones((len(self.identifiers)))/len(self.identifiers)
        self.h0 = torch.randn(1, 1, self.args.combined_dimensions)
        self.c0 = torch.randn(1, 1, self.args.combined_dimensions)

    def forward(self, data, graph, features, node):
        self.state, node, attention_score = self.step_block(data, graph, features, node)
        output, (self.h0, self.c0) = self.recurrent_block(self.state, (self.h0, self.c0))
        label_predictions, attention = self.down_block(output)
        self.step_block.attention = attention.view(-1)
        return F.log_softmax(label_predictions, dim=1), node, attention_score

def create_batches(graphs, batch_size):
    """
    :param graphs:
    :param batch_size:
    :return batches:
    """
    batches = [graphs[i:i + batch_size] for i in range(0, len(graphs), batch_size)]
    return batches

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

class GAMTrainer(object):

    def __init__(self, args):
        self.args = args
        self.GAM = GAM(args)
        self.setup_graphs()
        self.setup_logs()

    def setup_logs(self):
        self.logs = create_logs(self.args)

    def setup_graphs(self):
        self.training_graphs = glob.glob(self.args.train_graph_folder + "*.json")
        self.test_graphs = glob.glob(self.args.test_graph_folder + "*.json")

    def calculate_reward(self, target, predictions):
        reward = (target == torch.argmax(predictions))
        reward = 2*(reward.float()-0.5)
        return reward

    def calculate_predictive_loss(self, data, predictions):
        target = [data["target"]]
        target = torch.tensor(target)
        prediction_loss = F.nll_loss(predictions, target)
        return target, prediction_loss

    def process_graph(self, graph_path, batch_loss):
        data = json.load(open(graph_path))
        graph, features = create_features(data, self.GAM.identifiers)
        node = random.choice(graph.nodes())
        attention_loss = 0
        for t in range(self.args.time):
            predictions, node, attention_score = self.GAM(data, graph, features, node)
            target, prediction_loss = self.calculate_predictive_loss(data, predictions)
            batch_loss = batch_loss + prediction_loss
            if t < self.args.time-2:
                attention_loss = attention_loss + (self.args.gamma**(self.args.time-t))*torch.log(attention_score)
        reward = self.calculate_reward(target, predictions)
        batch_loss = batch_loss-reward*attention_loss
        self.GAM.reset_attention()
        return batch_loss

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        batch_loss = 0
        for graph_path in batch:
            batch_loss = self.process_graph(graph_path, batch_loss)
        batch_loss.backward(retain_graph = True)
        self.optimizer.step()
        return batch_loss.item()

    def update_log(self):
        average_loss = self.epoch_loss/self.nodes_processed
        self.logs["losses"].append(average_loss)

    def fit(self):
        self.GAM.train()
        self.optimizer = torch.optim.Adam(self.GAM.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        epoch_range = trange(self.args.epochs, desc = "Epoch: ", leave=True)
        for epoch in epoch_range:
            random.shuffle(self.training_graphs)
            batches = create_batches(self.training_graphs, self.args.batch_size)
            self.epoch_loss = 0
            self.nodes_processed = 0
            batch_range = trange(len(batches))
            for batch in batch_range:
                self.epoch_loss = self.epoch_loss + self.process_batch(batches[batch])
                self.nodes_processed = self.nodes_processed + len(batches[batch])
                loss_score = round(self.epoch_loss/self.nodes_processed ,4)
                batch_range.set_description("(Loss=%g)" % loss_score)
            self.update_log()

    def score_graph(self, data, predictions):
        target = data["target"]
        is_it_right = (target == np.argmax(predictions.detach()))
        self.predictions.append(is_it_right)

    def score(self):
        print("\nScoring test set.\n")
        self.GAM.eval()
        self.predictions = []
        for data in tqdm(self.test_graphs):
            data = json.load(open(data))
            graph, features = create_features(data, self.GAM.identifiers)
            node = random.choice(graph.nodes())
            for time in range(self.args.time):
                prediction, node, _ = self.GAM(data, graph, features, node)
            self.score_graph(data, prediction)
            self.GAM.reset_attention()
        self.accuracy = float(np.mean(self.predictions))
        print("\nThe test set accuracy is: "+str(round(self.accuracy,4))+".\n")

    def save_predictions_and_logs(self):
        self.logs["test_accuracy"] = self.accuracy
        with open(self.args.log_path,"w") as f:
            json.dump(self.logs,f)
        self.output_data = pd.DataFrame([[self.test_graphs[i], self.predictions[i].item()] for i in range(len(self.test_graphs))], columns = ["graph_id","predicted_label"])
        self.output_data.to_csv(self.args.prediction_path, index=None)


class MemoryGAM(object):
    def __init__(self):
        print("Yolo")


class MemoryGAMTrainer(object):
    def __init__(self,args):
        self.args = args
        x = MemoryGAM()
    def fit(self):
        print("Model")
    def score(self):
        print("Score")
    def save_predictions_and_logs(self):
        print("Saved.")

           
                
                    
