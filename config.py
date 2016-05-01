import tensorflow as tf

from classes.model import Layer


class BaseConfig():
    TRAINING_DATA = './assignment/train_potus_by_county.csv'
    TESTING_DATA = './assignment/train_potus_by_county.csv'

    TARGET_LABEL = 'Winner'

    OUTFILES = {'targets': './targets.csv',
                'preprocessing_means': './preprocessing_means.csv',
                'preprocessing_stddevs': './preprocessing_stddevs.csv',
                'graph_def': './graph_def.bin',
                'performance': './performance.txt',
                'predictions': './predictions.txt'}

    # defaults to random (non-persistent) seed if None
    SEED = 47

    # defaults to all detected if 0
    NUM_CORES = 3

    VERBOSE = False


class Config(BaseConfig):
    def __init__(self, hyperparams, layers):
        self.HYPERPARAMS = hyperparams
        self.LAYERS = layers


class GridSearchConfig(BaseConfig):
    def __init__(self, hyperparam_grid, hidden_layer_grid):
        self.HYPERPARAM_GRID = hyperparam_grid
        self.HIDDEN_LAYER_GRID = hidden_layer_grid


########################################################################################

HYPERPARAM_GRID = {'learning_rate': [0.05, 0.01, 0.1],
                   # keep probability for dropout (1 for none)
                   'dropout': [0.5, 0.7, 1],
                   # lambda for L2 regularization (0 for none)
                   'lambda_l2_reg': [1E-5, 1E-4, 1E-3, 0],
                   'n_minibatch': [100],
                   'epochs': [100]}

HIDDEN_LAYER_GRID = {'activation': [tf.nn.relu],# tf.nn.tanh, tf.nn.sigmoid],
                     'hidden_nodes': [[14],
                                      [12],
                                      [10],
                                      [8],
                                      [10, 8]]}

HYPERPARAMS = {'learning_rate': 0.05,
               'dropout': 0.7,
               'lambda_l2_reg': 1E-5,
               'n_minibatch': 100,
               'epochs': 100}

ARCHITECTURE = [
    # input & output nodes will be sized by data shape
    Layer('input', None, None),
    Layer('hidden_1', 12, tf.nn.relu),
    #Layer('hidden_2', 10, tf.nn.relu),
    Layer('output', None, tf.nn.softmax)
    ]

#config = GridSearchConfig(HYPERPARAM_GRID, HIDDEN_LAYER_GRID)
config = Config(HYPERPARAMS, ARCHITECTURE)
