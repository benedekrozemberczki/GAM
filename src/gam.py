"""Graph Attention Mechanism."""

import glob
import json
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm, trange
from utils import calculate_reward, calculate_predictive_loss
from utils import read_node_labels, create_logs, create_features, create_batches

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
        self.theta_step_1 = torch.nn.Parameter(torch.Tensor(len(self.identifiers),
                                                            self.args.step_dimensions))

        self.theta_step_2 = torch.nn.Parameter(torch.Tensor(len(self.identifiers),
                                                            self.args.step_dimensions))

        self.theta_step_3 = torch.nn.Parameter(torch.Tensor(2*self.args.step_dimensions,
                                                            self.args.combined_dimensions))

        torch.nn.init.uniform_(self.theta_step_1, -1, 1)
        torch.nn.init.uniform_(self.theta_step_2, -1, 1)
        torch.nn.init.uniform_(self.theta_step_3, -1, 1)

    def sample_node_label(self, orig_neighbors, graph, features):
        """
        Sampling a label from the neighbourhood.
        :param original_neighbors: Neighbours of the source node.
        :param graph: NetworkX graph.
        :param features: Node feature matrix.
        :return label: Label sampled from the neighbourhood with attention.
        """
        neighbor_vector = torch.tensor([1.0 if n in orig_neighbors else 0.0 for n in graph.nodes()])
        neighbor_features = torch.mm(neighbor_vector.view(1, -1), features)
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
        orig_neighbors = set(nx.neighbors(graph, node))
        label = self.sample_node_label(orig_neighbors, graph, features)
        labels = list(set(orig_neighbors).intersection(set(inverse_labels[str(label)])))
        new_node = random.choice(labels)
        new_node_attributes = torch.zeros((len(self.identifiers), 1))
        new_node_attributes[label, 0] = 1.0
        attention_score = self.attention[label]
        return new_node_attributes, new_node, attention_score

    def forward(self, data, graph, features, node):
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
        feature_row, node, attention_score = self.make_step(node, graph, features,
                                                            data["labels"], data["inverse_labels"])

        hidden_attention = torch.mm(self.attention.view(1, -1), self.theta_step_1)
        hidden_node = torch.mm(torch.t(feature_row), self.theta_step_2)
        combined_hidden_representation = torch.cat((hidden_attention, hidden_node), dim=1)
        state = torch.mm(combined_hidden_representation, self.theta_step_3)
        state = state.view(1, 1, self.args.combined_dimensions)
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
        predictions = torch.mm(hidden_state.view(1, -1), self.theta_classification)
        attention = torch.mm(hidden_state.view(1, -1), self.theta_rank)
        attention = torch.nn.functional.softmax(attention, dim=1)
        return predictions, attention

class GAM(torch.nn.Module):
    """
    Graph Attention Machine class.
    """
    def __init__(self, args):
        """
        Initializing the machine.
        :param args: Arguments object.
        """
        super(GAM, self).__init__()
        self.args = args
        self.identifiers, self.class_number = read_node_labels(self.args)
        self.step_block = StepNetworkLayer(self.args, self.identifiers)
        self.recurrent_block = torch.nn.LSTM(self.args.combined_dimensions,
                                             self.args.combined_dimensions, 1)

        self.down_block = DownStreamNetworkLayer(self.args, self.class_number, self.identifiers)
        self.reset_attention()

    def reset_attention(self):
        """
        Resetting the attention and hidden states.
        """
        self.step_block.attention = torch.ones((len(self.identifiers)))/len(self.identifiers)
        self.lstm_h_0 = torch.randn(1, 1, self.args.combined_dimensions)
        self.lstm_c_0 = torch.randn(1, 1, self.args.combined_dimensions)

    def forward(self, data, graph, features, node):
        """
        Doing a forward pass on a graph from a given node.
        :param data: Data dictionary.
        :param graph: NetworkX graph.
        :param features: Feature tensor.
        :param node: Source node identifier.
        :return label_predictions: Label prediction.
        :return node: New node to move to.
        :return attention_score: Attention score on selected node.
        """
        self.state, node, attention_score = self.step_block(data, graph, features, node)
        lstm_output, (self.h0, self.c0) = self.recurrent_block(self.state,
                                                               (self.lstm_h_0, self.lstm_c_0))
        label_predictions, attention = self.down_block(lstm_output)
        self.step_block.attention = attention.view(-1)
        label_predictions = torch.nn.functional.log_softmax(label_predictions, dim=1)
        return label_predictions, node, attention_score

class GAMTrainer(object):
    """
    Object to train a GAM model.
    """
    def __init__(self, args):
        self.args = args
        self.model = GAM(args)
        self.setup_graphs()
        self.logs = create_logs(self.args)

    def setup_graphs(self):
        """
        Listing the training and testing graphs in the source folders.
        """
        self.training_graphs = glob.glob(self.args.train_graph_folder + "*.json")
        self.test_graphs = glob.glob(self.args.test_graph_folder + "*.json")

    def process_graph(self, graph_path, batch_loss):
        """
        Reading a graph and doing a forward pass on a graph with a time budget.
        :param graph_path: Location of the graph to process.
        :param batch_loss: Loss on the graphs processed so far in the batch.
        :return batch_loss: Incremented loss on the current batch being processed.
        """
        data = json.load(open(graph_path))
        graph, features = create_features(data, self.model.identifiers)
        node = random.choice(list(graph.nodes()))
        attention_loss = 0
        for t in range(self.args.time):
            predictions, node, attention_score = self.model(data, graph, features, node)
            target, prediction_loss = calculate_predictive_loss(data, predictions)
            batch_loss = batch_loss + prediction_loss
            if t < self.args.time-2:
                attention_loss += (self.args.gamma**(self.args.time-t))*torch.log(attention_score)
        reward = calculate_reward(target, predictions)
        batch_loss = batch_loss-reward*attention_loss
        self.model.reset_attention()
        return batch_loss

    def process_batch(self, batch):
        """
        Forward and backward propagation on a batch of graphs.
        :param batch: Batch if graphs.
        :return loss_value: Value of loss on batch.
        """
        self.optimizer.zero_grad()
        batch_loss = 0
        for graph_path in batch:
            batch_loss = self.process_graph(graph_path, batch_loss)
        batch_loss.backward(retain_graph=True)
        self.optimizer.step()
        loss_value = batch_loss.item()
        self.optimizer.zero_grad()
        return loss_value

    def update_log(self):
        """
        Adding the end of epoch loss to the log.
        """
        average_loss = self.epoch_loss/self.nodes_processed
        self.logs["losses"].append(average_loss)

    def fit(self):
        """
        Fitting a model on the training dataset.
        """
        print("\nTraining started.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.optimizer.zero_grad()
        epoch_range = trange(self.args.epochs, desc="Epoch: ", leave=True)
        for _ in epoch_range:
            random.shuffle(self.training_graphs)
            batches = create_batches(self.training_graphs, self.args.batch_size)
            self.epoch_loss = 0
            self.nodes_processed = 0
            batch_range = trange(len(batches))
            for batch in batch_range:
                self.epoch_loss = self.epoch_loss + self.process_batch(batches[batch])
                self.nodes_processed = self.nodes_processed + len(batches[batch])
                loss_score = round(self.epoch_loss/self.nodes_processed, 4)
                batch_range.set_description("(Loss=%g)" % loss_score)
            self.update_log()

    def score_graph(self, data, prediction):
        """
        Scoring the prediction on the graph.
        :param data: Data hash table of graph.
        :param prediction: Label prediction.
        """
        target = data["target"]
        is_it_right = (target == prediction)
        self.predictions.append(is_it_right)

    def score(self):
        """
        Scoring the test set graphs.
        """
        print("\n")
        print("\nScoring the test set.\n")
        self.model.eval()
        self.predictions = []
        for data in tqdm(self.test_graphs):
            data = json.load(open(data))
            graph, features = create_features(data, self.model.identifiers)
            node_predictions = []
            for _ in range(self.args.repetitions):
                node = random.choice(list(graph.nodes()))
                for _ in range(self.args.time):
                    prediction, node, _ = self.model(data, graph, features, node)
                node_predictions.append(np.argmax(prediction.detach()))
                self.model.reset_attention()
            prediction = max(set(node_predictions), key=node_predictions.count)
            self.score_graph(data, prediction)
        self.accuracy = float(np.mean(self.predictions))
        print("\nThe test set accuracy is: "+str(round(self.accuracy, 4))+".\n")

    def save_predictions_and_logs(self):
        """
        Saving the predictions as a csv file and logs as a JSON.
        """
        self.logs["test_accuracy"] = self.accuracy
        with open(self.args.log_path, "w") as f:
            json.dump(self.logs, f)
        cols = ["graph_id", "predicted_label"]
        predictions = [[self.test_graphs[i], self.predictions[i].item()] for i in range(len(self.test_graphs))]
        self.output_data = pd.DataFrame(predictions, columns=cols)
        self.output_data.to_csv(self.args.prediction_path, index=None)
