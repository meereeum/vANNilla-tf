from __future__ import division
import itertools
from collections import namedtuple
import sys
import re

import numpy as np
import tensorflow as tf
import pandas as pd

########################################################################################

# DATA I/O

class DataIO:
    def __init__(self, df, target_label, norm_fn = None, clip_to = None):
        """Data class with functions for preprocessing and streaming

        Args: df (pandas DataFrame)
              target_label (string)
              norm_fn (function) - optional function for features normalization
              clip_to (iterable) - optional 2-tuple of max/min values for clipping
        """
        self.regex = {'features': '^[^({})]'.format(target_label),
                      'targets': '^{}'.format(target_label)}

        self.df = self.normalizeXs(df, norm_fn) if norm_fn else df
        self.df = self.encodeYs(target_label)
        if clip_to:
            # don't touch encoded one-hots
            assert sum(abs(x) > 1 for x in clip_to)
            self.df = self.df.clip(*clip_to)

    @property
    def len_(self):
        return len(self.df)

    @property
    def n_features(self):
        x, _ = self.splitXY()
        return x.shape[1]

    def splitXY(self, df = None):
        """Split given DataFrame into DataFrames representing features & targets"""
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
        return (df - mean) / std

    @staticmethod
    def minMax(df):
        """Center dataframe column ranges to [-1, 1]"""
        max_, min_ = df.max(axis=0), df.min(axis=0)
        midrange = (max_ + min_) / 2
        half_range = (max_ - min_) / 2
        return (df - midrange) / half_range

    @staticmethod
    def centerMeanAndNormalize(df):
        """Center dataframe column means to 0 and scale range to [-1,1]"""
        return minMax(df - df.mean(axis=0))


########################################################################################

# ARTIFICIAL NEURAL NET

class Model():
    def __init__(self, hyperparams = None, layers = None, graph_def = None, seed = None):
        """Artificial neural net with architecture according to given layers,
        training according to given hyperparameters

        Args: hyperparams (dictionary) - model hyperparameters
              layers (list) - sequential Layer objects
              graph_def (filepath) - model binary to restore
              seed (int) - optional random seed to persist random variable
                initialization repeatably across sessions
        """
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        if hyperparams and layers:
            # turn off dropout, l2 reg, max epochs if not specified
            DEFAULTS = {'dropout': 1, 'lambda_l2_reg': 0, 'epochs': np.inf}
            self.hyperparams = DEFAULTS
            self.hyperparams.update(**hyperparams)

            self.layers = layers

            (self.x, self.y, self.dropout, self.accuracy, self.cost,
             self.train_op) = self._buildGraph()

        elif graph_def:
            # restore from previously saved Graph
            with open(graph_def, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            self.x, self.dropout, self.predictions = tf.import_graph_def(
                graph_def, name = '', return_elements =
                ['x:0', 'dropout:0', 'accuracy/predictions:0'])

        else:
            raise(ValueError, 'Must supply hyperparameters and layer architecture to initialize Model, or supply graph definition to restore previous Model graph')

    @staticmethod
    def crossEntropy(observed, actual):
        """Cross-entropy between two equally sized tensors

        Returns: tensorflow scalar (i.e. averaged over minibatch)
        """
        # bound values by clipping to avoid nan
        with tf.name_scope('cross_entropy'):
            return -tf.reduce_mean(actual*tf.log(tf.clip_by_value(observed, 1e-10, 1.0)))

    def _buildGraph(self):
        """Build TensorFlow graph representing neural net with desired architecture +
        training ops for feed-forward and back-prop to minimize cost function"""
        x_in = tf.placeholder(tf.float32, shape=[None, # None dim enables variable batch size
                                                 self.layers[0].nodes], name='x')
        xs = [x_in]

        def wbVars(nodes_in, nodes_out, scope):
            """Helper to initialize trainable weights & biases"""
            #with tf.name_scope(scope):
            initial_w = tf.truncated_normal([nodes_in, nodes_out],
                                            #stddev = (2/nodes_in)**0.5)
                                            stddev = nodes_in**-0.5, name =
                                            '{}/truncated_normal'.format(scope))
            initial_b = tf.random_normal([nodes_out], name =
                                         '{}/random_normal'.format(scope))

            return (tf.Variable(initial_w, trainable=True, name='{}/weights'.format(scope)),
                    tf.Variable(initial_b, trainable=True, name='{}/biases'.format(scope)))

        #ws_and_bs = (wbVars(in_.nodes, out.nodes)
                     #for in_, out in zip(self.layers, self.layers[1:]))
        ws_and_bs = [wbVars(in_.nodes, out.nodes, out.name)
                     for in_, out in zip(self.layers, self.layers[1:])]

        dropout = tf.placeholder(tf.float32, name='dropout')

        #for i, layer in enumerate(self.layers[1:]):
            #with tf.name_scope(layer.name):
                #w, b = ws_and_bs.next()
                ## add dropout to hidden but not input weights
                #if i > 0:
                    #w = tf.nn.dropout(w, dropout)
                #xs.append(layer.activation(tf.nn.xw_plus_b(xs[i], w, b)))
        for i, layer in enumerate(self.layers[1:]):
            w, b = ws_and_bs[i]
            with tf.name_scope(layer.name):
                # add dropout to hidden but not input weights
                if i > 0:
                    w = tf.nn.dropout(w, dropout)
                xs.append(layer.activation(tf.nn.xw_plus_b(xs[i], w, b)))

        # use identity to set explicit name for output node
        y_out = tf.identity(xs[-1], name='y_out')
        y = tf.placeholder(tf.float32, shape=[None, 2], name='y')

        # cost & training
        with tf.name_scope('l2_regularization'):
            lmbda = tf.constant(self.hyperparams['lambda_l2_reg'], tf.float32,
                                name='lambda')
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w, _ in ws_and_bs])
            weighted_l2_loss = tf.mul(lmbda, l2_loss)

        cost = tf.add(weighted_l2_loss, Model.crossEntropy(y_out, y), name='cost')
        train_op = tf.train.AdamOptimizer(self.hyperparams['learning_rate'])\
                                          .minimize(cost)
        #tvars = tf.trainable_variables()
        #grads = tf.gradients(cost, tvars)
        # TODO: cap gradients ? learning rate decay ?
        #train_op = tf.train.GradientDescentOptimizer(self.hyperparams['learning_rate'])\
                           #.apply_gradients(zip(grads, tvars))
        with tf.name_scope('accuracy'):
            predictions = tf.argmax(y_out, 1, name = 'predictions')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,
                                                       tf.argmax(y, 1)), tf.float32))

        return (x_in, y, dropout, accuracy, cost, train_op)

    def kFoldTrain(self, data, k = 10, verbose = False, num_cores = None):
        """Train model using k-fold cross-validation of full set of training data"""
        # highest cross-validation accuracy per fold
        self.best_cross_vals = []
        # earliest epoch corresponding to highest cross-val accuracy per fold
        self.best_stopping_epochs = []
        # list of lists of each cross-validation accuracy per epoch, across all folds
        self.l_cross_vals = []

        STREAM_KWARGS = {'train': {'batchsize': self.hyperparams['n_minibatch'],
                                   'max_iter': self.hyperparams['epochs']},
                         'validate': {}}

        for train, validate in data.kFoldCrossVal(k):
            # train on (k-1) folds
            streams = {k: data.stream(v, **STREAM_KWARGS[k])
                       for k, v in (('train', train), ('validate', validate))}

            cross_vals = self.train(streams, len(train), verbose = verbose,
                                    num_cores = num_cores)

            self.l_cross_vals.append(cross_vals)
            self.best_cross_vals.append(self.record_cross_val)
            self.best_stopping_epochs.append(self.record_epoch)

        assert len(self.l_cross_vals) == k

    def train(self, data_dict, n_train, verbose = False, num_cores = 0,
              save = False, outfile = './graph_def',
              logging = False, logdir = './log'):
        """Train on training data and, if supplied, cross-validate accuracy of
        validation data at every epoch.

        Args: data_dict (dictionary) - containing `train` (&, optionally, `validate`)
                key/s + corresponding iterable streams of (x, y) tuples
              n_train (int) - size of training data
              verbose (bool) - print to STDOUT to monitor cost/accuracy
              num_cores (int) - number of cores to use for TensorFlow operations
                 (defaults to all cores detected automatically)
              save (bool) - freeze & save trained model binary
              outfile (filepath) - optional path/to/file of model binary

        Returns: list of cross-validation accuracies (empty if no `validate` data)
        """
        iters_per_epoch = (n_train // self.hyperparams['n_minibatch']) + \
                          ((n_train % self.hyperparams['n_minibatch']) != 0)

        validate = 'validate' in data_dict.iterkeys()
        cross_vals = []

        config = tf.ConfigProto(inter_op_parallelism_threads = num_cores,
                                intra_op_parallelism_threads = num_cores)

        with tf.Session(config = config) as sesh:
            sesh.run(tf.initialize_all_variables())

            #if logging:
            logger = tf.train.SummaryWriter(logdir, sesh.graph_def)

            #while len(cross_vals) < self.hyperparams['epochs']:
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

                    if validate and i % iters_per_epoch == 0:
                        # cross-validate with leftout fold
                        x, y = data_dict['validate'].next()
                        feed_dict = {self.x: x, self.y: y,
                                    self.dropout: 1.0} # keep prob 1
                        accuracy = sesh.run(self.accuracy, feed_dict)
                        cross_vals.append(accuracy)

                        if verbose:
                            print 'CROSS VAL accuracy at epoch {}: {}'.format(
                                i // iters_per_epoch, accuracy)

            except(KeyboardInterrupt):
                sys.exit("""

Epochs: {}
Current cross-val accuracies: {}
                """.format(i / iters_per_epoch, cross_vals))

            if save:
                self._freeze()
                with open(outfile, 'wb') as f:
                    f.write(sesh.graph_def.SerializeToString())

            #if logging:
            logger.flush()
                #logger.close()

        if validate:
            assert len(cross_vals) == self.hyperparams['epochs']
            # find earliest epoch corresponding to best cross-validation accuracy
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

        return cross_vals

    def _freeze(self):
        """Add nodes to assign weights & biases to constants containing current
        trained values, enabling them to be saved by TensorFlow's graph_def"""
        regex = re.compile('^[^:]*') # string up to first `:`
        with tf.name_scope('assign_ops'):
            for tvar in tf.trainable_variables():
                tf.assign(tvar, tvar.eval(),
                          name = re.match(regex, tvar.name).group(0))


########################################################################################

# Grid search: tune hyperparameters and neural net hidden layer architecture

Layer = namedtuple('Layer', ('name', 'nodes', 'activation'))

def combinatorialGridSearch(d_hyperparam_grid, d_layer_grid):
    """Generate all combinations of hyperparameter and layer configs

    Args: d_hyperparam_grid (dictionary) - hyperparameter names (string)
            as keys + corresponding potential settings (list) as values
          d_layer_grid (dictionary) - containing the following items...
            'activation': list of tensorflow activation functions, &
            'hidden_nodes': list of lists of hidden layer architectures,
               i.e. nodes per layer per config

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
        # name scope hidden layers as `hidden_1`, `hidden_2`, etc
        itertools.izip(['hidden_{}'.format(i + 1) for i in xrange(max_depth)],
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
            'model_params': './model_params.txt',
            'graph_def': './graph_def.bin',
            'performance': './performance.txt'}

HYPERPARAM_GRID = {'learning_rate': [0.01, 0.05, 0.1],
                   # keep probability for dropout (1 for none)
                   'dropout': [0.5, 0.7, 1],
                   # lambda for L2 regularization (0 for none)
                   'lambda_l2_reg': [0, 1E-5, 1E-4, 1E-3],
                   'n_minibatch': [100],
                   'epochs': [100]}

HIDDEN_LAYER_GRID = {'activation': [tf.nn.relu, tf.nn.sigmoid],# tf.nn.tanh],
                     'hidden_nodes': [[10],
                                      [10, 7],
                                      [10, 10],
                                      [10, 7, 7]]}

SEED = 47

NUM_CORES = 3


########################################################################################

def doWork_combinatorial(file_in = TRAINING_DATA, target_label = TARGET_LABEL,
                         d_hyperparams = HYPERPARAM_GRID,
                         d_architectures = HIDDEN_LAYER_GRID,
                         seed = None, num_cores = None, verbose = False):
    """Preprocess training data, use grid search of all combinatorial possibilities
    for given hyperparameters and layer architecture to tune artificial neural net
    model, and save files necessary for resurrection of tuned model"""
    df = pd.read_csv(file_in)
    assert sum(df.isnull().any()) == False

    # record categorical targets for decoding test set
    targets = list(pd.get_dummies(df[target_label]))
    with open(OUTFILES['targets'], 'w') as f:
        f.write(','.join(targets))

    # preprocess features
    data = DataIO(df, target_label, DataIO.gaussianNorm, [-10, 10])

    # extract raw features mean, stddev from test set to use for all preprocessing
    raw_features, _ = DataIO(df, target_label).splitXY()
    params = (raw_features.mean(axis=0), raw_features.std(axis=0))
    for k, param in zip(('preprocessing_means', 'preprocessing_stddevs'), params):
         with open(OUTFILES[k], 'w') as f:
             param.to_csv(f)

    # tune hyperparameters, architecture
    combos = combinatorialGridSearch(d_hyperparams, d_architectures)
    overall_best_mean, overall_std = 0, 0

    for hyperparams, architecture in combos:
        architecture[0] = architecture[0]._replace(nodes = data.n_features)
        architecture[-1] = architecture[-1]._replace(nodes = len(targets))

        model = Model(hyperparams, architecture, seed = seed)
        model.kFoldTrain(data, k = 10, verbose = verbose, num_cores = num_cores)

        print """
HYPERPARAMS: {}
LAYERS:
    {}
MAX CROSS-VAL ACCURACIES: {}
AT EPOCHS: {}
        """.format(model.hyperparams,
                   '\n    '.join(str(l) for l in model.layers),
                   model.best_cross_vals, model.best_stopping_epochs)

        # find epoch with best mean cross-val accuracy across all k folds of training
        mean_accs = [np.mean(accs) for accs in itertools.izip(*model.l_cross_vals)]
        i, best_mean = max(enumerate(mean_accs), key = lambda x: x[1])

        # accuracies at `best` epoch across k folds
        selected_accs = [cross_vals[i] for cross_vals in model.l_cross_vals]
        std = np.std(selected_accs)

        if best_mean > overall_best_mean or (best_mean == overall_best_mean and
                                             std < overall_std):
            overall_best_mean, overall_std = best_mean, std
            stopping_epoch = i + 1 # list of accuracies starts after epoch 1
            accs = selected_accs
            best_model = (hyperparams, architecture)
            print """
New best model!
Accuracies at epoch {}: {}
Mean: {} +/- {}
            """.format(stopping_epoch, accs, overall_best_mean, overall_std)

    hyperparams, architecture = best_model

    print """
BEST HYPERPARAMS!... {}
BEST ARCHITECTURE!... {}
median = {}
mean = {}""".format(hyperparams, architecture, np.median(accs), overall_best_mean)

    with open(OUTFILES['performance'], 'w') as f:
        f.write("""k-fold cross-validation accuracies at selected epoch:
    {}

Mean: {}
STDEV: {}
Median: {}
IQR: {}
Range: {}
        """.format('\n    '.join(str(a) for a in accs),
                    overall_best_mean, overall_std, np.median(accs),
                    np.subtract(*np.percentile(accs, [75, 25])), np.ptp(accs)))

    hyperparams['epochs'] = stopping_epoch

    with open(OUTFILES['model_params'], 'w') as f:
        f.write("""
HYPERPARAMS:
    {}
ARCHITECTURE:
    {}
        """.format(hyperparams, '\n    '.join(str(l) for l in architecture)))

    datastream = {'train': data.stream(batchsize = hyperparams['n_minibatch'],
                                       max_iter = stopping_epoch)}

    model = Model(hyperparams, architecture, seed = seed)
    model.train(datastream, data.len_, num_cores = num_cores, verbose = True,
                save = True, outfile = OUTFILES['graph_def'], logging = True)
    print "Trained model saved to: {}".format(OUTFILES['graph_def'])


########################################################################################

if __name__ == '__main__':
    doWork_combinatorial(seed = SEED, num_cores = NUM_CORES)
