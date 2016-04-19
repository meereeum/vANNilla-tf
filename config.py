import tensorflow as tf

TRAINING_DATA = './assignment/train_potus_by_county.csv'
TESTING_DATA = './assignment/train_potus_by_county.csv'

TARGET_LABEL = 'Winner'

OUTFILES = {'targets': './targets.csv',
            'preprocessing_means': './preprocessing_means.csv',
            'preprocessing_stddevs': './preprocessing_stddevs.csv',
            #'model_params': './model_training_params.txt',
            'graph_def': './graph_def.bin',
            'performance': './performance.txt',
            'predictions': './predictions.txt'}

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

SEED = 47

NUM_CORES = 3

VERBOSE = False
