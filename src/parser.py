import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it learns on the Erdos-Renyi dataset.
    The default hyperparameters give good results without cross-validation.
    """

    parser = argparse.ArgumentParser(description = "Run GAM.")

    parser.add_argument('--test-graph-path',
                        nargs = '?',
                        default = './input/erdos_multi_class/test/',
	                help = 'Erdos datasets.')

    parser.add_argument('--train-graph-path',
                        nargs = '?',
                        default = './input/erdos_multi_class/train/',
	                help = 'Erdos datasets.')

    parser.add_argument('--log-path',
                        nargs = '?',
                        default = './logs/erdos_gam_logs.json',
	                help = 'Log json.')

    parser.add_argument('--epochs',
                        type = int,
                        default = 10,
	                help = 'Number of training epochs. Default is 100.')

    parser.add_argument('--step-dimensions',
                        type = int,
                        default = 32,
	                help = 'Number of SVD feature extraction dimensions. Default is 64.')

    parser.add_argument('--combined-dimensions',
                        type = int,
                        default = 64,
	                help = 'Number of SVD feature extraction dimensions. Default is 64.')

    parser.add_argument('--batch-size',
                        type = int,
                        default = 32,
	                help = 'Number of graphs processed per batch. Default is 32.')

    parser.add_argument('--time',
                        type = int,
                        default = 20,
	                help = 'Random seed for sklearn pre-training. Default is 100.')

    parser.add_argument('--aggents',
                        type = int,
                        default = 10,
	                help = 'Random seed for sklearn pre-training. Default is 10.')

    parser.add_argument('--gamma',
                        type = float,
                        default = 0.99,
	                help = 'Embedding regularization parameter. Default is 1.0.')

    parser.add_argument('--learning-rate',
                        type = float,
                        default = 0.001,
	                help = 'Learning rate. Default is 0.001.')

    parser.add_argument('--weight-decay',
                        type = float,
                        default = 10**-5,
	                help = 'Learning rate. Default is 10^-5.')
    
    return parser.parse_args()

