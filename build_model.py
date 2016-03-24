from __future__ import division
from collections import namedtuple
import itertools

import numpy as np
import tensorflow as tf
import pandas as pd

########################################################################################

# DATA I/O

class DataIO:
    def __init__(self, df, target_label, norm_fn = None, clip_to = None, k = 10):
        """ Data object with functions for preprocessing and streaming

        Args: df (pandas dataframe)
              target_label (string corresponding to target label)
              norm_fn (optional, function for features normalization)
              clip_to (optional, iterable of max/min values for clipping)
        """
        self.regex = {'features': '^[^({})]'.format(target_label),
                      'targets': '^{}'.format(target_label)}

        self.encodeYs(target_label)

        self.df = self.normalizeXs(df, norm_fn) if norm_fn else df
        if clip_to:
            self.df = self.df.clip(*clip_to) # TODO: check for >1 so as not to affect targets ?

        self.k = k

    @property
    def len_(self):
        return len(self.df)

    @property
    def n_features(self):
        x, _ = self.splitXY()
        return x.shape[1]

    def kFoldCrossVal(self):
        # shuffle
        df = self.df.sample(frac=1)
        # split into k pieces
        df_arr = [ df[i::self.k] for i in xrange(self.k)]
        assert len(df_arr) == self.k

        for i, val in enumerate(df_array):
            #validate = pd.DataFrame(df_array[i])
            validate = df_array[i]
            assert type(validate) == pd.DataFrame
            print 'validate', validate

            train_list = df_array[:i] + df_array[i+1:]
            print 'training', train_list

            return (train_list, validate)

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
        return pd.get_dummies(self.df, columns=[target_str])

    @staticmethod
    def stream(df, batchsize = None, max_iter = np.inf):
        """Generator of minibatches of given batch size, optionally
        limited by maximum numer of iterations over dataset (=epochs).

        Yields: (x,y) tuples of numpy arrays representing features & targets
        """
        len_ = len(df)
        batchsize = len_ if not batchsize else batchsize
        reps = 0
        while True:
            if reps <= max_iter:
                for i in xrange(0, df.len_, batchsize):
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


def splitTrainValidate(df, perc_training = 0.8):
    """Split dataframe into training & validation sets based on given %"""
    train = df.sample(frac=perc_training)#, random_state=200)
    validate = df.drop(train.index)
    return (train, validate)



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
            w = tf.nn.dropout(w, dropout) if i > 0 else w
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

    def train(self, data, verbose = False,
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
        cross_validate = ('validate' in data_dict.iterkeys())

        n_train = data_dict['train'].len_
        iters_per_epoch = (n_train // self.hyperparams['n_minibatch']) + \
                          ((n_train % self.hyperparams['n_minibatch']) != 0)

        train_list, validate = DataIO.kFoldCrossVal(data)

        if save_best:
            saver = tf.train.saver() # defaults to saving max most recent 5 checkpoints

        with tf.Session() as sesh:
            sesh.run(tf.initialize_all_variables())
            # TODO: logger

            epochs = 0
            cross_vals = []
            try:
                for i, (x, y) in enumerate(datastream['train']):
                    feed_dict = {self.x: x, self.y: y,
                                 self.dropout: self.hyperparams['dropout']}
                    accuracy, cost, _ = sesh.run([self.accuracy, self.cost,
                                                  self.train_op], feed_dict)

                    if verbose and i % 20 == 0:
                        print 'cost at iteration {}: {}'.format(i, cost)
                        print 'accuracy: {}'.format(accuracy)

                    if cross_validate and i % iters_per_epoch == 0:
                        epochs += 1
                        x, y = datastream['validate'].next()
                        feed_dict = {self.x: x, self.y: y,
                                     self.dropout: 1.0} # keep prob 1
                        accuracy = sesh.run(self.accuracy, feed_dict)

                        if save_best and accuracy > max(cross_vals):
                            saver.save(sesh, outfile, global_step = epochs)
                            if log_perf:
                                with open(outfile_perf, 'a') as f:
                                    performance = ['{}_{}'.format(outfile, epochs),
                                                   accuracy]
                                    f.write()

                        cross_vals.append(accuracy)

                        if verbose:
                            print 'CROSS VAL accuracy at epoch {}: {}'.format(epochs, accuracy)
            except(KeyboardInterrupt):
                pass
            finally:
                # TODO: close logger
                if cross_vals:
                    i, max_ = max(enumerate(cross_vals), key = lambda x: x[1])
                    self.max_cross_val_accuracy = max_
                    if verbose:
                        print """
HYPERPARAMS: {}
LAYERS:
    {}
MAX CROSS-VAL ACCURACY (at epoch {}): {}
                        """.format(self.hyperparams,
                                   '\n    '.join(str(l) for l in self.layers),
                                   i+1, self.max_cross_val_accuracy)

    # def _validate(self, session):
    #     x, y = self.datastream['validate'].next()
    #     feed_dict = {self.x: x, self.y: y, self.dropout: 1.0} # keep prob 1
    #     accuracy, cost = session.run([self.accuracy, self.cost], feed_dict)
    #     return (accuracy, cost)


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

    # preprocess features
    data = DataIO(train, target_label, gaussianNorm, [-10,10], k = 10)

    # extract raw features mean, stddev from test set to use for all preprocessing
    params = (data.mean(axis=0), data.std(axis=0))
    # for k, param in zip(('preprocessing_means', 'preprocessing_stddevs'), params):
    #     with open(OUTFILES[k], 'w') as f:
    #         param.to_csv(f)

    combos = combinatorialGridSearch(d_hyperparams, d_architectures)

    record_cross_val_acc = 0
    acc_mean = 0

    # tune hyperparameters, architecture
    for hyperparams, architecture in combos:
        architecture[0] = architecture[0]._replace(nodes = data['train'].n_features)
        architecture[-1] = architecture[-1]._replace(nodes = len(targets))

        model = Model(hyperparams, architecture)

        accuracies = []
        for i in xrange(3):
            model.train(data)
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
median = {}
mean = {}
    """.format(hyperparams, architecture, record_cross_val_acc, acc_mean)

    hyperparams['epochs'] = 300
    model = Model(hyperparams, architecture)
    for i in xrange(5):
        model.train(data, save_best = True, outfile =
                    '{}_{}'.format(OUTFILES['model_binary'], i))
    # TODO: log performance to performance.txt

########################################################################################

if __name__ == "__main__":
    doWork_combinatorial()
