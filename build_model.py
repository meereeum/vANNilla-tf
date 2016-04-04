from __future__ import division
import itertools
from collections import namedtuple
import sys
import json

import numpy as np
import tensorflow as tf
import pandas as pd

########################################################################################

# DATA I/O

class DataIO:
    def __init__(self, df, target_label, norm_fn = None, clip_to = None):
        """ Data object with functions for preprocessing and streaming

        Args: df (pandas dataframe)
              target_label (string corresponding to target label)
              norm_fn (optional, function for features normalization)
              clip_to (optional, iterable of max/min values for clipping)
        """
        self.regex = {'features': '^[^({})]'.format(target_label),
                      'targets': '^{}'.format(target_label)}

        self.df = self.normalizeXs(df, norm_fn) if norm_fn else df
        self.df = self.encodeYs(target_label)
        if clip_to:
            self.df = self.df.clip(*clip_to) # TODO: check for >1 so as not to affect targets ?

    @property
    def len_(self):
        return len(self.df)

    @property
    def n_features(self):
        x, _ = self.splitXY()
        return x.shape[1]

    def splitXY(self, df = None):
        """Split given dataframe into dataframes representing features & targets"""
        df = df if isinstance(df, pd.DataFrame) else self.df # <-- default
        return tuple(df.filter(regex = self.regex[k])
                     for k in ('features', 'targets'))

    def normalizeXs(self, df, norm_fn):
        """Normalize features by given function"""
        xs, ys = self.splitXY(df)
        return pd.concat([norm_fn(xs), ys], axis=1)

    def encodeYs(self, target_str):
        """Encode categorical values (labeled with given target_str) as one-hots

        Returns: dataframe including binary {0, 1} columns for unique labels
        """
        try:
            encoded = pd.get_dummies(self.df, columns=[target_str])
        except(ValueError):
            # ignore test data with no targets - but don't fail silently
            print """
Warning: categorical values not encoded...no targets labeled `{}` found""".format(target_str)
            encoded = self.df
        return encoded

    def kFoldCrossVal(self, k):
        """TODO
        """
        # shuffle
        df = self.df.sample(frac=1)
        # chunk into k folds
        #fold_idxs = [range(i, self.len_, k) for i in xrange(k)]
        df_arr = [df[i::k] for i in xrange(k)]

        for i, validate in enumerate(df_arr):
            train_list = df_arr[:i] + df_arr[i+1:]
            train = pd.concat(train_list, axis=0)

            assert len(train) + len(validate) ==  len(df)
            yield (train, validate)

    def stream(self, df = None, batchsize = None, max_iter = np.inf):
        """Generator of minibatches of given batch size, optionally
        limited by maximum numer of iterations over dataset (=epochs).

        Yields: (x,y) tuples of numpy arrays representing features & targets
        """
        df = df if isinstance(df, pd.DataFrame) else self.df # <-- default
        len_ = len(df)
        batchsize = len_ if not batchsize else batchsize
        reps = 0
        while True:
            if reps < max_iter:
                for i in xrange(0, len_, batchsize):
                    try:
                        batched = df[i:i+batchsize]
                    except(IndexError):
                        batched = df[i:]
                    finally:
                        x, y = self.splitXY(batched)
                        yield (x.values, y.values)
            else:
                break

            reps += 1

    @staticmethod
    def gaussianNorm(df, mean = None, std = None):
        """Normalize dataframe columns by z-score (mean 0, stddev 1) if params unspecified.
        Else, center by the given mean & normalize by the given stddev."""
        mean = mean if isinstance(mean, pd.Series) else df.mean(axis=0)
        std = std if isinstance(std, pd.Series) else df.std(axis=0)
        return (df - mean)/std

    @staticmethod
    def minMax(df):
        """Center dataframe column ranges to [-1, 1]"""
        max_, min_ = df.max(axis=0), df.min(axis=0)
        midrange = (max_ + min_)/2
        half_range = (max_ - min_)/2
        return (df - midrange)/half_range

    @staticmethod
    def centerMeanAndNormalize(df):
        """Center dataframe column means to 0 and scale range to [-1,1]"""
        return minMax(df - df.mean(axis=0))


########################################################################################

# ARTIFICIAL NEURAL NET

class Model():
    def __init__(self, hyperparams, layers):
        """Artificial neural net with architecture according to given layers,
        training according to given hyperparameters.

        Args: hyperparams (dictionary of model hyperparameters)
              layers (dictionary of Layer objects)
        """
        # turn off dropout, l2 reg, max epochs if not specified
        DEFAULTS = {'dropout': 1, 'lambda_l2_reg': 0, 'epochs': np.inf}
        self.hyperparams = DEFAULTS
        self.hyperparams.update(**hyperparams)

        self.layers = layers

        self.x, self.y, self.dropout, self.accuracy, self.cost, self.train_op = self.buildGraph()

    @staticmethod
    def crossEntropy(observed, actual):
        """Cross-entropy between two equally sized tensors

        Returns: tensorflow scalar (i.e. averaged over minibatch)
        """
        # bound values by clipping to avoid nan
        return -tf.reduce_mean(actual*tf.log(tf.clip_by_value(observed, 1e-10, 1.0)))

    def buildGraph(self):
        """Build tensorflow graph representing neural net with desired architecture +
        training ops for feed-forward and back-prop to minimize given cost function"""
        x_in = tf.placeholder(tf.float32, shape=[None, # None dim enables variable batch size
                                                 self.layers[0].nodes], name='x')
        xs = [x_in]

        def wbVars(nodes_in, nodes_out, scope):
            """Helper to initialize trainable weights & biases"""
            initial_w = tf.truncated_normal([nodes_in, nodes_out],
                                            #stddev = (2/nodes_in)**0.5)
                                            stddev = nodes_in**-0.5)
            initial_b = tf.random_normal([nodes_out])
            #initial_b = tf.zeros([nodes_out]) # TODO: test me!
            with tf.name_scope(scope):
                return (tf.Variable(initial_w, trainable=True, name='weights'),
                        tf.Variable(initial_b, trainable=True, name='biases'))

        ws_and_bs = [wbVars(in_.nodes, out.nodes, out.name)
                     for in_, out in zip(self.layers, self.layers[1:])]

        dropout = tf.placeholder(tf.float32, name='dropout')
        for i, layer in enumerate(self.layers[1:]):
            w, b = ws_and_bs[i]
            # add dropout to hidden weights but not input layer
            w = tf.nn.dropout(w, dropout) if i > 0 else w #TODO: leave ??
            #w_ = tf.nn.dropout(w, dropout) if i > 0 else w #TODO: leave ??
            xs.append(layer.activation(tf.nn.xw_plus_b(xs[i], w, b)))

        # cost & training
        y_out = xs[-1]
        y = tf.placeholder(tf.float32, shape=[None, 2], name='y')

        cost = Model.crossEntropy(y_out, y)
        lmbda = tf.constant(self.hyperparams['lambda_l2_reg'], tf.float32)
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w, _ in ws_and_bs])
        cost += tf.mul(lmbda, l2_loss, name='l2_regularization')

        train_op = tf.train.AdamOptimizer(self.hyperparams['learning_rate']).minimize(cost)
        #tvars = tf.trainable_variables()
        #grads = tf.gradients(cost, tvars)
        # TODO: cap gradients ? learning rate decay ?
        #train_op = tf.train.GradientDescentOptimizer(self.hyperparams['learning_rate'])\
                           #.apply_gradients(zip(grads, tvars))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_out, 1),
                                                   tf.argmax(y, 1)), tf.float32))

        return (x_in, y, dropout, accuracy, cost, train_op)

    def kFoldTrain(self, data, k = 10, verbose = False):

        self.best_cross_vals = []
        self.best_stopping_epochs = []

        for train, validate in data.kFoldCrossVal(k):
            # train on (k-1) folds
            n_train = len(train)
            train = data.stream(train, batchsize = self.hyperparams['n_minibatch'],
                                       max_iter = self.hyperparams['epochs'])
            validate = data.stream(validate)
            self.train({'train': train, 'validate': validate}, n_train, verbose = verbose)

            self.best_cross_vals.append(self.record_cross_val)
            self.best_stopping_epochs.append(self.record_epoch)

        assert len(self.best_cross_vals) == k

    def train(self, data_dict, n_train, verbose = False,
              save_best = False, outfile = './model_chkpt',
              log_perf = False, outfile_perf = './performance.txt'):
        """Train on training data and, if supplied, cross-validate accuracy of
        validation data at every epoch.

        Args: data_dict (dictionary with 'train' (&, optionally, 'validate') key/s
                 + corresponding DataIO object/s)
              verbose (optional bool for monitoring cost/accuracy)
              save_best (optional bool to save model/s with highest cross-validation
                accuracy to disk) # TODO: best ?
              outfile (optional path/to/file for model checkpoints)
              log_perf (optional bool to save performance predictions to file)
              outfile_perf (optional path/to/file for model performance values)
        """
        #n_train = len(data_dict['train'].next())
        iters_per_epoch = (n_train // self.hyperparams['n_minibatch']) + \
                          ((n_train % self.hyperparams['n_minibatch']) != 0)

        cross_vals = []
        with tf.Session() as sesh:
            sesh.run(tf.initialize_all_variables())
            while len(cross_vals) < self.hyperparams['epochs']:
                try:
                    for i, (x, y) in enumerate(data_dict['train'], 1):
                        # train
                        feed_dict = {self.x: x, self.y: y,
                                    self.dropout: self.hyperparams['dropout']}
                        accuracy, cost, _ = sesh.run([self.accuracy, self.cost,
                                                    self.train_op], feed_dict)

                        if verbose and i % 50 == 0:
                            print 'cost after iteration {}: {}'.format(i + 1, cost)
                            print 'accuracy: {}'.format(accuracy)

                        if i % iters_per_epoch == 0:
                            # cross-validate with leftout fold
                            x, y = data_dict['validate'].next()
                            feed_dict = {self.x: x, self.y: y,
                                        self.dropout: 1.0} # keep prob 1
                            accuracy = sesh.run(self.accuracy, feed_dict)

                            cross_vals.append(accuracy)

                            if verbose:
                                print 'CROSS VAL accuracy at epoch {}: {}'.format(
                                    i//iters_per_epoch, accuracy)


                except(KeyboardInterrupt):
                    sys.exit("""

Epochs: {}
Current cross-val accuracies: {} """.format(i/iters_per_epoch, cross_vals))

        assert len(cross_vals) == self.hyperparams['epochs']
        # finds lowest i corresponding to highest cross_vals value
        i, max_ = max(enumerate(cross_vals, 1), key = lambda x: x[1])
        self.record_epoch = i
        self.record_cross_val = max_

        if verbose:
            print """
HYPERPARAMS: {}
LAYERS:
{}
MAX CROSS-VAL ACCURACY (at epoch {}): {}
            """.format(self.hyperparams,
                        '\n    '.join(str(l) for l in self.layers),
                        i, max_)


########################################################################################

# Grid search: tune hyperparameters and neural net hidden layer architecture

Layer = namedtuple('Layer', ('name', 'nodes', 'activation'))

def combinatorialGridSearch(d_hyperparam_grid, d_layer_grid):
    """Generate all combinations of hyperparameter and layer configs

    Args: d_hyperparam_grid (dictionary with hyperparameter names (strings)
            as keys + corresponding lists of settings as values)
          d_layer_grid (dictionary with the following items...
            'activation': list of tensorflow activation functions, &
            'hidden_nodes': list of lists of hidden layer architectures,
            i.e. nodes per layer)

    Returns: generator yielding tuples of dictionaries (hyperparams, architecture)
    """
    DEFAULT = {'n_minibatch': 100,
               'epochs': 200}
    def merge_with_default(new_params):
        """Update DEFAULT dict with passed hyperparams --
        any DEFAULT keys duplicated by new_params will be replaced
        """
        d = DEFAULT.copy()
        d.update(new_params)
        return d

    # generate lists of tuples of (k, v) pairs for each hyperparameter + value
    hyperparam_tuples = (itertools.izip(itertools.repeat(k),v)
                         for k,v in d_hyperparam_grid.iteritems())
    # all combinatorial hyperparameter settings
    HYPERPARAMS = (merge_with_default(params) for params in
                   itertools.product(*hyperparam_tuples))

    max_depth = max(len(layers) for layers in d_layer_grid['hidden_nodes'])
    # generate nested tuples describing layer names, # nodes, functions
    # per hidden layer of each set of layers describing an architecture
    layer_tuples = (
        # name scope hidden layers as 'hidden_1', 'hidden_2', ...
        itertools.izip(['hidden_{}'.format(i+1) for i in xrange(max_depth)],
                       layers, itertools.repeat(fn))
        for layers, fn in itertools.product(d_layer_grid['hidden_nodes'],
                                            d_layer_grid['activation'])
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


########################################################################################

TRAINING_DATA = './assignment/train_potus_by_county.csv'

TARGET_LABEL = 'Winner'

OUTFILES = {'targets': './targets.csv',
            'preprocessing_means': './preprocessing_means.csv',
            'preprocessing_stddevs': './preprocessing_stddevs.csv',
            'train': './data_training_cleaned.csv',
            'validate': './data_validation_cleaned.csv',
            'model_binary': './model_bin',
            'performance': './performance.txt'}

HYPERPARAM_GRID = {'learning_rate': [0.01, 0.05, 0.1],
                   # keep probability for dropout (1 for none)
                   'dropout': [0.5, 0.7, 1],#[0.3, 0.5, 0.7, 1],
                   # lambda for L2 regularization (0 for none)
                   'lambda_l2_reg': [0, 1E-5, 1E-4],# 1E-3],#[0, 1E-5, 1E-4, 1E-3],
                   'n_minibatch': [100],
                   'epochs': [100]}

HIDDEN_LAYER_GRID = {'activation': [tf.nn.relu],#, tf.nn.sigmoid, tf.nn.tanh],
                     'hidden_nodes': [[10],
                                      #[10, 7],
                                      [10, 10],
                                      [10, 7, 7]]}


########################################################################################

def doWork_combinatorial(file_in = TRAINING_DATA, target_label = TARGET_LABEL,
                         d_hyperparams = HYPERPARAM_GRID,
                         d_architectures = HIDDEN_LAYER_GRID):
    df = pd.read_csv(file_in)
    assert sum(df.isnull().any()) == False

    # record categorical targets for decoding test set
    targets = list(pd.get_dummies(df[target_label]))
    # with open(OUTFILES['targets'], 'w') as f:
    #     f.write(','.join(targets))

    # preprocess features
    data = DataIO(df, target_label, DataIO.gaussianNorm, [-10, 10])

    # extract raw features mean, stddev from test set to use for all preprocessing
    params = (data.df.mean(axis=0), data.df.std(axis=0))
    # for k, param in zip(('preprocessing_means', 'preprocessing_stddevs'), params):
    #     with open(OUTFILES[k], 'w') as f:
    #         param.to_csv(f)

    combos = combinatorialGridSearch(d_hyperparams, d_architectures)

    record_cross_val_acc = 0
    acc_mean = 0

    # tune hyperparameters, architecture
    for hyperparams, architecture in combos:
        architecture[0] = architecture[0]._replace(nodes = data.n_features)
        architecture[-1] = architecture[-1]._replace(nodes = len(targets))

        model = Model(hyperparams, architecture)
        model.kFoldTrain(data, k = 10, verbose = False)

        #accuracies = []
        #for i in xrange(3):
            #model.kFold_train(data, k = 10)
            ##accuracies.append(model.max_cross_val_accuracy)

        print """
HYPERPARAMS: {}
LAYERS:
    {}
MAX CROSS-VAL ACCURACIES: {}
AT EPOCHS: {}
        """.format(model.hyperparams,
                   '\n    '.join(str(l) for l in model.layers),
                   model.best_cross_vals, model.best_stopping_epochs)

        median = np.median(model.best_cross_vals)
        mean = np.mean(model.best_cross_vals)
        if median > record_cross_val_acc or \
        (median == record_cross_val_acc and mean > acc_mean):
            record_cross_val_acc, acc_mean = median, mean
            best_model = (hyperparams, architecture)

    hyperparams, architecture = best_model

    print """
BEST HYPERPARAMS!... {}
BEST ARCHITECTURE!... {}
median = {}
mean = {}
    """.format(hyperparams, architecture, record_cross_val_acc, acc_mean)

    #hyperparams['epochs'] = 300
    #model = Model(hyperparams, architecture)
    #for i in xrange(5):
        #model.train(data)
    # TODO: log performance to performance.txt
    ##model_params = (hyperparams, architecture)
    #for k, param in zip(('hyperparams', 'architecture'), best_model):#model_params):
        #with open(OUTFILES[k], 'w' as f:
                  #json.dump(param, f, indent = 4)

    #with open(OUTFILES['hyperparams'], 'w') as f:
        #json.dump(hyperparams, f, indent = 4)
    #with open(OUTFILES['architecture'], 'w') as f:
        #to_dump = [layer._asdict() for layer in architecture]
        #json.dump(to_dump, f, indent = 4)


########################################################################################

if __name__ == "__main__":
    doWork_combinatorial()
