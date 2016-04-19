from collections import namedtuple
import itertools

import numpy as np

from model import Model


Layer = namedtuple('Layer', ('name', 'nodes', 'activation'))

########################################################################################

class GridSearch():
    """Tune hyperparameters and neural netowrk hidden layer architecture"""
    def __init__(self, d_hyperparam_grid, d_layer_grid):
        """Implement grid search to fine-tune hyperparams and layer architecture
        according to cross-validation values of associated neural nets

        Args: d_hyperparam_grid (dictionary) - hyperparameter names (string)
                as keys + corresponding potential settings (list) as values
              d_layer_grid (dictionary) - containing the following items...
                'activation': list of tensorflow activation functions, &
                'hidden_nodes': list of lists of hidden layer architectures,
                i.e. nodes per layer per config
        """
        DEFAULTS = {'learning_rate': [0.05],
                    'dropout': [1], # no dropout - i.e. keep prob 1
                    'lambda_l2_reg': [0], # no L2 regularization
                    'n_minibatch': [100],
                    'epochs': [200]}
        self.d_hyperparam_grid = DEFAULTS.copy()
        # any DEFAULT items with key duplicated by d_hyperparam_grid will be replaced
        self.d_hyperparam_grid.update(**d_hyperparam_grid)

        self.d_layer_grid = d_layer_grid

    def iterCombos(self):
        """Generate all combinations of hyperparameter and layer configs

        Returns: generator yielding tuple(dictionary, list of Layers)
          corresponding, respectively, to hyperparams, architecture
        """
        # yields generators of tuples of (k, v) pairs for each hyperparameter + value
        hyperparam_tuples = (itertools.izip(itertools.repeat(k), v)
                             for k, v in self.d_hyperparam_grid.iteritems())
        # all combinatorial hyperparameter settings
        HYPERPARAMS = (dict(params) for params in
                       itertools.product(*hyperparam_tuples))

        max_depth = max(len(layers) for layers in self.d_layer_grid['hidden_nodes'])
        # generate nested tuples describing layer names, # nodes, functions
        # per hidden layer of each set of layers describing an architecture
        layer_tuples = (
            # name scope hidden layers as `hidden_1`, `hidden_2`, etc
            itertools.izip(['hidden_{}'.format(i + 1) for i in xrange(max_depth)],
                        layers, itertools.repeat(fn))
            for layers, fn in itertools.product(self.d_layer_grid['hidden_nodes'],
                                                self.d_layer_grid['activation'])
        )
        # all combinatorial nn architectures of Layers
        ARCHITECTURES = (
            # input & output nodes will be sized by data shape
            [Layer('input', None, None)] +
            [Layer(*tup) for tup in hidden_architecture] +
            [Layer('output', None, tf.nn.softmax)]
            for hidden_architecture in layer_tuples
        )

        return itertools.product(HYPERPARAMS, ARCHITECTURES)

    def tuneParams(self, data, k = 10, seed = None, num_cores = None,
                   verbose = False):
        """Tune hyperparameters, architecture using k-fold cross-validation of
        input `data` (DataIO object)

        Returns: tuple(dict, list of Layers) corresponding to hyperparams
          and layer architecture with highest mean cross-validation accuracy across
          all `k` folds
        """
        overall_best_mean, overall_std = 0, 0
        best_accs = []

        for hyperparams, architecture in self.iterCombos():
            architecture[0] = architecture[0]._replace(nodes = data.n_features)
            architecture[-1] = architecture[-1]._replace(nodes = data.n_targets)

            model = Model(hyperparams, architecture)
            results = model.kFoldTrain(data, k = k, verbose = verbose,
                                       num_cores = num_cores, seed = seed)

            print """
HYPERPARAMS: {}
LAYERS:
    {}
MAX CROSS-VAL ACCURACIES: {}
AT EPOCHS: {}
""".format(model.hyperparams, '\n    '.join(str(l) for l in model.layers),
           results['best_cross_vals'], results['best_stopping_epochs'])

            best_cross_vals = results['best_cross_vals']
            mean = np.mean(best_cross_vals)
            std = np.std(best_cross_vals)
            if mean > overall_best_mean or (
                    mean == overall_best_mean and std < overall_std):
                overall_best_mean, overall_std = mean, std
                best_accs = best_cross_vals
                best_model = (hyperparams, architecture)
                print """
New best model!
Mean: {} +/- {}
""".format(overall_best_mean, overall_std)

        print """
BEST HYPERPARAMS!... {}
BEST ARCHITECTURE!... {}
median = {}
mean = {}""".format(hyperparams, architecture, np.median(best_accs), overall_best_mean)

        return best_model # (hyperparams, architecture)
