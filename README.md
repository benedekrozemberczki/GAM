GAM
============================================
A PyTorch implementation of "Graph Classification using Structural Attention" (KDD 2018),

<div style="text-align:center"><img src ="attention_true.jpg" ,width=600/></div>
<p align="justify">
Graph classification is a problem with practical applications in many different domains. To solve this problem, one usually calculates certain graph statistics (i.e. , graph features) that help discriminate between graphs of different classes. When calculating such features, most existing approaches process the entire graph. In a graphlet-based approach, for instance, the entire graph is processed to get the total count of different graphlets or subgraphs. In many real-world applications, however, graphs can be noisy with discriminative patterns confined to certain regions in the graph only. In this work, we study the problem of attention-based graph classification . The use of attention allows us to focus on small but informative parts of the graph, avoiding noise in the rest of the graph. We present a novel RNN model, called the Graph Attention Model (GAM), that processes only a portion of the graph by adaptively selecting a sequence of “informative” nodes. Experimental results on multiple real-world datasets show that the proposed method is competitive against various well-known methods in graph classification even though our method is limited to only a portion of the graph.</p>

This repository provides an implementation for SGCN as described in the paper:

> Graph Classification using Structural Attention.
> John Boaz Lee, Ryan Rossi, and Xiangnan Kong
> KDD, 2018.
> [[Paper]](http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf)


### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx           1.11
tqdm               4.28.1
numpy              1.15.4
pandas             0.23.4
texttable          1.5.0
argparse           1.1.0
sklearn            0.20.0
torch              1.0.0.
torchvision        0.2.1
```
### Datasets

The code takes graphs for training from an input folder where each graph is stored as a JSON. Graphs used for testing are also stored as JSON files. Every node id, node label and class has to be indexed from 0.

These JSON files have the following key-value structure:

```javascript
{"target": 1,
 "edges": [[0, 1], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
 "labels": {"0": 2, "1": 3, "2": 2, "3": 3, "4": 4},
 "inverse_labels": {"2": [0, 2], "3": [1, 3], "4": [4]}}
```
The **target key** has an integer value, which is the ID of the target class (e.g. Carcinogenicity). The **edges key** has an edge list value for the graph of interest. The **labels key** has a dictonary value for each node, these labels are stored as key-value pairs (e.g. node - atom pair). The **inverse_labels key** has a key for each node label and the values are lists containing the nodes that have a specific node label.

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path                STR    Input graph path.          Default is `input/bitcoin_otc.csv`.
  --features-path            STR    Membership path.           Default is `input/bitcoin_otc.csv`.
  --embedding-path           STR    Embedding path.            Default is `output/embedding/bitcoin_otc_sgcn.csv`.
  --regression-weights-path  STR    Regression weights path.   Default is `output/weights/bitcoin_otc_sgcn.csv`.
  --log-path                 STR    Log path.                  Default is `logs/bitcoin_otc_logs.json`.  
```

#### Model options

```
  --epochs                INT         Number of SGCN training epochs.      Default is 100. 
  --reduction-iterations  INT         Number of SVD epochs.                Default is 128.
  --reduction-dimensions  INT         SVD dimensions.                      Default is 30.
  --seed                  INT         Random seed value.                   Default is 42.
  --lamb                  FLOAT       Embedding regularization parameter.  Default is 1.0.
  --test-size             FLOAT       Test ratio..                         Default is False.  
  --learning-rate         FLOAT       Learning rate.                       Default is 0.001.  
  --weight-decay          FLOAT       Weight decay.                        Default is 10^-5. 
  --layers                LST         Layer sizes in model.                Default is [64, 32].
  --spectral-features     BOOL        Layer sizes in autoencoder model.    Default is True
  --general-features      BOOL        Loss calculation for the model.      Sets spectral features to False.  
```

### Examples

The following commands learn a node embedding, regression weights and write the embedding to disk. The node representations are ordered by the ID. The layer sizes can be set manually.

Creating an SGCN embedding of the default dataset. Saving the embedding, regression weights and logs at default paths.
```
python src/main.py
```
<p align="center">
<img style="float: center;" src="sgcn_run_example.jpg">
</p>

Creating an SGCN model of the default dataset with a 96-64-32 architecture.
```
python src/main.py --layers 96 64 32
```
Creating a single layer SGCN model with 32 features.
```
python src/main.py --layers 32
```
Creating an embedding with some custom learning rate and epoch number.
```
python src/main.py --learning-rate 0.001 --epochs 200
```
Creating an embedding of another dataset with features present a signed `Erdos-Renyi` graph. Saving the weight output and logs in a custom folder.
```
python src/main.py --general-features --edge-path input/erdos_renyi_edges.csv --features-path input/erdos_renyi_features.csv --embedding-path output/embedding/erdos_renyi.csv --regression-weights-path output/weights/erdos_renyi.csv --log-path logs/erdos_renyi.json
```

