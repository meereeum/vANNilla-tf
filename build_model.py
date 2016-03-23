from __future__ import division
from collections import namedtuple
import itertools

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

    def stream(self, batchsize = None, max_iter = np.inf):
        """Generator of minibatches of given batch size, optionally
        limited by maximum numer of iterations over dataset (=epochs).

        Yields: (x,y) tuples of numpy arrays representing features & targets
        """
        batchsize = self.len_ if not batchsize else batchsize
        reps = 0
        while True:
            if reps <= max_iter:
                # shuffle
                self.df = self.df.sample(frac=1).reset_index(drop=True)
                for i in xrange(0, self.len_, batchsize):
                    try:
                        batched = self.df[i:i+batchsize]
                    except(IndexError):
                        batched = self.df[i:]
                    finally:
                        x, y = self.splitXY(batched)
                        yield (x.values, y.values)
            else:
                break

            reps += 1


def splitTrainValidate(df, perc_training = 0.8):
    """Split dataframe into training and validation sets based on given %"""
    train = df.sample(frac=perc_training)#, random_state=200)
    validate = df.drop(train.index)
    return (train, validate)

def encodeYs(df, target_str):
    """Encode categorical values (labeled with given target_str) as one-hots

    Returns: dataframe including binary {0,1} columns for unique categories
    """
    return pd.get_dummies(df, columns=[target_str])

def gaussianNorm(arr, mean = None, std = None):
    """Normalize dataframe columns by z-score (mean 0, STD 1) if no mean/stddev specified.
    Else, center by the given mean and normalize by the given stddev."""
    mean = mean if isinstance(mean, pd.Series) else arr.mean(axis=0)
    std = std if isinstance(std, pd.Series) else arr.std(axis=0)
    return (arr - mean)/std

def minMax(arr):
    """Center dataframe or array columns to be within [-1, 1]"""
    max_, min_ = arr.max(axis=0), arr.min(axis=0)
    midrange = (max_ + min_)/2
    half_range = (max_ - min_)/2
    return (arr - midrange)/half_range

def centerMeanAndNormalize(arr):
    """Center mean to 0 and scale range to [-1,1]"""
    return minMax(arr - arr.mean(axis=0))

########################################################################################

# ARTIFICIAL NEURAL NET

class Model():
    def __init__(self, data_dict, hyperparams, layers, cross_validate=False, verbose=False):
        """Artificial neural net with architecture according to given layers,
        training according to given hyperparameters.

        Args: data_dict (dictionary with 'train' (and, optionally, 'validate') key/s
        and corresponding DataIO object/s

        hyperparams (dictionary of model hyperparameters)

        layers (dictionary of Layer objects)

        cross_validate (bool)
        """
        if 'validate' not in data_dict.iterkeys() and cross_validate:
            raise ValueError('Must include validation dataset in order to cross-validate')
        self.layers = layers
        self.cross_validate = cross_validate
        self.verbose = verbose

        self.hyperparams = hyperparams
        # turn off dropout, l2 reg, max epochs if not specified
        DEFAULTS = {'dropout': 1, 'lambda_l2_reg': 0, 'epochs': np.inf}
        #unspecified_k = DEFAULTS.viewkeys() - self.hyperparams.viewkeys()
        unspecified = ((k, v) for k, v in DEFAULTS.iteritems()
                       if k not in self.hyperparams.iterkeys())
        self.hyperparams.update(unspecified)

        n_train = data_dict['train'].len_
        self.iters_per_epoch = (n_train // self.hyperparams['n_minibatch']) + \
                               (n_train % self.hyperparams['n_minibatch'] != 0)

        stream_kwargs = {'train': {'batchsize': self.hyperparams['n_minibatch'],
                                   'max_iter': self.hyperparams['epochs']},
                         'validate': dict()}
        self.datastream = {k: v.stream(**stream_kwargs[k])
                           for k,v in data_dict.iteritems()}

        self.x, self.y, self.dropout, self.accuracy, self.cost, self.train_op = self.buildGraph()

    def train(self):
        """Train on training data and cross-validate with accuracy of validation data at every epoch"""
        with tf.Session() as sesh:
            sesh.run(tf.initialize_all_variables())
            # TODO: logger

            epochs = 0
            cross_vals = []
            try:
                for i, (x, y) in enumerate(self.datastream['train']):
                    feed_dict = {self.x: x, self.y: y,
                                 self.dropout: self.hyperparams['dropout']}
                    accuracy, cost, _ = sesh.run([self.accuracy, self.cost,
                                                  self.train_op], feed_dict)

                    if self.verbose and i % 20 == 0:
                        print 'cost at iteration {}: {}'.format(i.cost)
                        print 'accuracy: {}'.format(accuracy)

                    if self.cross_validate and i % self.iters_per_epoch == 0:
                        x, y = self.datastream['validate'].next()
                        feed_dict = {self.x: x, self.y: y,
                                     self.dropout: 1.0} # keep prob 1
                        accuracy = sesh.run(self.accuracy, feed_dict)
                        cross_vals.append(accuracy)
                        epochs += 1

                        if self.verbose and accuracy > 0.85:
                            print 'CROSS VAL accuracy at epoch {}: {}'.format(epochs, accuracy)
            except(KeyboardInterrupt):
                pass
            finally:
                # TODO: close logger
                if self.cross_validate:
                    self.max_cross_val_accuracy = max(cross_vals)

                    if self.verbose:
                        print """
HYPERPARAMS: {}
LAYERS:
    {}
MAX CROSS-VAL ACCURACY: {}
                        """.format(self.hyperparams,
                                '\n    '.join(str(l) for l in self.layers),
                                accuracies)

    # def _validate(self, session):
    #     x, y = self.datastream['validate'].next()
    #     feed_dict = {self.x: x, self.y: y, self.dropout: 1.0} # keep prob 1
    #     accuracy, cost = session.run([self.accuracy, self.cost], feed_dict)
    #     return (accuracy, cost)

    def buildGraph(self):#, cost_fn = crossEntropy()):
        """Build tensorflow graph representing neural net with desired architecture
        and training ops for feed-forward & back-prop"""
        x_in = tf.placeholder(tf.float32, shape=[None, # None dim enables variable sized batches
                                                 self.layers[0].nodes], name='x')
        xs = [x_in]

        def wbVars(nodes_in, nodes_out, scope):
            """Helper to initialize trainable weights and biases as tf.Variables"""
            initial_w = tf.truncated_normal([nodes_in, nodes_out],
                                            stddev = nodes_in**-0.5)
            initial_b = tf.random_normal([nodes_out])
            with tf.name_scope(scope):
                return (tf.Variable(initial_w, trainable=True, name='weights'),
                        tf.Variable(initial_b, trainable=True, name='biases'))

        ws_and_bs = [wbVars(in_.nodes, out.nodes, out.name)
                     for in_, out in zip(self.layers, self.layers[1:])]

        dropout = tf.placeholder(tf.float32, name='dropout')
        for i, layer in enumerate(self.layers[1:]):
            w, b = ws_and_bs[i]
            # add dropout to hidden weights but not input
            if self.hyperparams['dropout'] and i > 0:
                w = tf.nn.dropout(w, dropout)
            xs.append(layer.activation(tf.nn.xw_plus_b(xs[i], w, b)))

        # cost & training
        y_out = xs[-1]
        y = tf.placeholder(tf.float32, shape=[None, 2], name='y')

        cost = self.crossEntropy(y_out, y) # TODO: cost_fn with default ?

        if self.hyperparams['lambda_l2_reg']:
            lmbda = tf.constant(self.hyperparams['lambda_l2_reg'], tf.float32)
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w, _ in ws_and_bs])
            cost += tf.mul(lmbda, l2_loss, name='l2_regularization')

        train_op = tf.train.AdamOptimizer(self.hyperparams['learning_rate']).minimize(cost)
        #tvars = tf.trainable_variables()
        #grads = tf.gradients(cost, tvars)
        # TODO: cap gradients ? learning rate decay ?
        #train_op = tf.train.GradientDescentOptimizer(self.hyperparams['learning_rate'])\
                           #.apply_gradients(zip(grads, tvars))

        #correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y_out, 1),
                                                 #tf.argmax(y, 1)), tf.uint32))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_out, 1),
                                                   tf.argmax(y, 1)), tf.float32))

        return (x_in, y, dropout, accuracy, cost, train_op)

    @staticmethod
    def crossEntropy(observed, actual):
        """Cross-entropy between two equally sized tensors

        Returns: tensorflow scalar (i.e. averaged over minibatch)
        """
        # bound values by clipping to avoid nan
        return -tf.reduce_mean(actual*tf.log(tf.clip_by_value(observed, 1e-10, 1.0)))


########################################################################################

# Grid search: tune hyperparameters and neural net hidden layer architecture

Layer = namedtuple('Layer', ('name', 'nodes', 'activation'))

def combinatorialGridSearch(d_hyperparam_grid, d_layer_grid):
    """
    Args: d_hyperparam_grid (dictionary with hyperparameter names (strings)
            as keys + corresponding lists of settings as values)
          d_layer_grid (dictionary with the following items...
            'activation': list of potential activation functions, &
            'hidden_nodes': list of lists of hidden layer architectures
            i.e. nodes per layer)
    """
    DEFAULT = {'n_minibatch': 100,
               'epochs': 200}

    def merge_with_default(new_params):
        """Update DEFAULT dict with passed hyperparams.
        Any DEFAULT keys duplicated by new_params will be replaced.
        """
        d = DEFAULT.copy()
        d.update(new_params)
        return d

    # generate lists of tuples of (k,v) items for each hyperparameter value
    hyperparam_tuples = (itertools.izip(itertools.repeat(k),v)
                         for k,v in d_hyperparam_grid.iteritems())
    # all combinatorial hyperparameter settings
    HYPERPARAMS = (merge_with_default(params) for params in
                   itertools.product(*hyperparam_tuples))

    max_depth = max(len(layers) for layers in d_layer_grid['hidden_nodes'])
    # generate nested tuples describing layer names, # nodes, functions
    # per hidden layer of each combinatorial nn hidden architecture
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

HYPERPARAM_GRID = {'learning_rate': [0.05, 0.1],#0.05, 0.1],
                   # keep probability for dropout (1 for none)
                   'dropout': [0.5, 0.7],#, 1],#[0.3, 0.5, 0.7, 1],
                   # lambda for L2 regularization (0 for none)
                   'lambda_l2_reg': [1E-4, 1E-3],#[0, 1E-5, 1E-4, 1E-3],
                   'n_minibatch': [100],
                   'epochs': [100]}

HIDDEN_LAYER_GRID = {'activation': [tf.nn.relu],#, tf.nn.sigmoid, tf.nn.tanh],
                     'hidden_nodes': [[10],
                                      #[10, 7],
                                      [10, 10],
                                      [10, 7, 7]]}

# HYPERPARAMS = {'learning_rate': 0.01,
#                'n_minibatch': 100,
#                'dropout': 0.5, # keep probability for dropout (1 for none)
#                'lambda_l2_reg': 0.001, # lamba for L2 regularization (0 for none)
#                #'epochs': 200}
#                }

# ARCHITECTURE = [
#     Layer('input', None, None), # input & output nodes will be sized by data shape
#     Layer('hidden_1', 10, tf.nn.relu),
#     Layer('hidden_2', 7, tf.nn.relu),
#     #Layer('hidden_3', 7, tf.nn.relu),
#     Layer('output', None, tf.nn.softmax),
# ]

########################################################################################

# def doWork(file_in = TRAINING_DATA, target_label = TARGET_LABEL,
#            hyperparams = HYPERPARAMS, architecture = ARCHITECTURE):
#     df = pd.read_csv(file_in)
#     assert sum(df.isnull().any()) == False

#     # record categorical targets for decoding test set
#     targets = list(pd.get_dummies(df[target_label]))
#     with open(OUTFILES['targets'], 'w') as f:
#         f.write(','.join(targets))

#     df = encodeYs(df, target_label)
#     train, validate = splitTrainValidate(df, perc_training=0.8)

#     data = {k: DataIO(df, target_label, gaussianNorm)#, [-5.0,5.0])
#             for k, df in (('train', train), ('validate', validate))}

#     # for k, v in data.iteritems():
#     #     with open(OUTFILES[k], 'w') as f:
#     #         f.write(v.df.to_csv(index=False))

#     architecture[0] = architecture[0]._replace(nodes = data['train'].n_features)
#     architecture[-1] = architecture[-1]._replace(nodes = len(targets))

#     model = Model(data, hyperparams, architecture, cross_validate = True)
#     model.train()


def doWork_combinatorial(file_in = TRAINING_DATA, target_label = TARGET_LABEL,
                         d_hyperparams = HYPERPARAM_GRID,
                         d_architectures = HIDDEN_LAYER_GRID):
    df = pd.read_csv(file_in)
    assert sum(df.isnull().any()) == False

    # record categorical targets for decoding test set
    targets = list(pd.get_dummies(df[target_label]))
    # with open(OUTFILES['targets'], 'w') as f:
    #     f.write(','.join(targets))

    df = encodeYs(df, target_label)
    train, validate = splitTrainValidate(df, perc_training=0.8)

    # extract raw features mean, stddev from test set to use for all preprocessing
    raw_x, _ = DataIO(train, target_label).splitXY()
    params = (raw_x.mean(axis=0), raw_x.std(axis=0))
    # for k, param in zip(('preprocessing_means', 'preprocessing_stddevs'), params):
    #     with open(OUTFILES[k], 'w') as f:
    #         param.to_csv(f)

    # preprocess features
    data = {k: DataIO(dataset, target_label, lambda x: gaussianNorm(x, *params))
            for k, dataset in (('train', train), ('validate', validate))}
    # for k, v in data.iteritems():
    #     with open(OUTFILES[k], 'w') as f:
    #         v.df.to_csv(f, index=False)

    # tune hyperparameters, architecture
    combos = combinatorialGridSearch(d_hyperparams, d_architectures)

    record_cross_val_acc = 0
    acc_mean = 0
    for hyperparams, architecture in combos:
        architecture[0] = architecture[0]._replace(nodes = data['train'].n_features)
        architecture[-1] = architecture[-1]._replace(nodes = len(targets))

        #model = Model(data, hyperparams, architecture, cross_validate=True) # TODO: outside of loop ? so reset datastream
        accuracies = []
        for i in xrange(3):
            model = Model(data, hyperparams, architecture, cross_validate=True)
            model.train()
            accuracies.append(model.max_cross_val_accuracy)

        print """
HYPERPARAMS: {}
LAYERS:
    {}
MAX CROSS-VAL ACCURACIES: {}
        """.format(model.hyperparams,
                   '\n    '.join(str(l) for l in model.layers),
                   accuracies)

        median = np.median(accuracies)
        mean = np.mean(accuracies)
        if median > record_cross_val_acc or \
        (median == record_cross_val_acc and mean > acc_mean):
            record_cross_val_acc, acc_mean = median, mean
            best_model = (hyperparams, architecture)

    hyperparams, architecture = best_model
    print """
BEST HYPERPARAMS!... {}
BEST ARCHITECTURE!... {}
    """.format(hyperparams, architecture)
    #model = Model(data, hyperparams, architecture, cross_validate=True)
    #model.train()

########################################################################################

if __name__ == "__main__":
    doWork_combinatorial()
