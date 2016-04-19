from __future__ import division
import re

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, hyperparams = None, layers = None, graph_def = None):
        """Artificial neural network with architecture according to given layers,
        training according to given hyperparameters

        Args: hyperparams (dictionary) - model hyperparameters
              layers (list) - sequential Layer objects
              graph_def (filepath) - model binary to restore
        """
        tf.reset_default_graph()

        if hyperparams and layers:
            # turn off dropout, l2 reg, max epochs if not specified
            DEFAULTS = {'dropout': 1, 'lambda_l2_reg': 0, 'epochs': np.inf}
            self.hyperparams = DEFAULTS
            self.hyperparams.update(**hyperparams)

            self.layers = layers

            (self.x, self.y, self.dropout, self.accuracy, self.cost,
             self.train_op) = self._buildGraph()

        elif graph_def:
            # restore from previously saved tf.Graph
            with open(graph_def, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            self.x, self.dropout, self.predictions = tf.import_graph_def(
                graph_def, name = '', return_elements =
                ['x:0', 'dropout:0', 'accuracy/predictions:0'])

        else:
            raise(ValueError,
                  ('Must supply hyperparameters and layer architecture to initialize'
                  'Model, or supply graph definition to restore saved Model'))

    @staticmethod
    def crossEntropy(observed, actual):
        """Cross-entropy between two equally sized tensors

        Returns: tensorflow scalar (i.e. averaged over minibatch)
        """
        with tf.name_scope('cross_entropy'):
            # bound values by clipping to avoid nan
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

            return (tf.Variable(initial_w, trainable=True,
                                name='{}/weights'.format(scope)),
                    tf.Variable(initial_b, trainable=True,
                                name='{}/biases'.format(scope)))

        ws_and_bs = [wbVars(in_.nodes, out.nodes, out.name)
                     for in_, out in zip(self.layers, self.layers[1:])]

        dropout = tf.placeholder(tf.float32, name='dropout')

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
        with tf.name_scope('accuracy'):
            predictions = tf.argmax(y_out, 1, name = 'predictions')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,
                                                       tf.argmax(y, 1)), tf.float32))

        return (x_in, y, dropout, accuracy, cost, train_op)

    def kFoldTrain(self, data, k = 10, verbose = False, num_cores = None, seed = None):
        """Train model using k-fold cross-validation of full set of training data"""
        results = {'best_cross_vals': [], # highest cross-val accuracy (per fold)
                   'best_stopping_epochs': [], # corresponding epoch (per fold)
                   'l_cross_vals': []} # nested list of cross-val accs over all epochs, all folds

        STREAM_KWARGS = {'train': {'batchsize': self.hyperparams['n_minibatch'],
                                   'max_iter': self.hyperparams['epochs']},
                         'validate': {}}

        for train, validate in data.kFoldCrossVal(k):
            # train on (k-1) folds, cross-validate on kth
            streams = {k: data.stream(v, **STREAM_KWARGS[k])
                       for k, v in (('train', train), ('validate', validate))}

            cross_vals = self.train(streams, len(train), verbose = verbose,
                                    num_cores = num_cores, seed = seed)

            i, max_ = max(enumerate(cross_vals, 1), key = lambda x: (x[1], x[0]))

            results['l_cross_vals'] += [cross_vals]
            results['best_cross_vals'] += [max_]
            results['best_stopping_epochs'] += [i]

        assert len(results['l_cross_vals']) == k
        return results

    def train(self, data_dict, n_train, seed = None, verbose = False,
              num_cores = 0, save = False, outfile = './graph_def',
              logging = False, logdir = './log'):
        """Train on training data and, if supplied, cross-validate accuracy of
        validation data at every epoch. Optionally, save trained values (at each
        epoch for which the validation accuracy is >= the previous record accuracy,
        or the last training epoch if no `validate` data suplied).

        Args: data_dict (dictionary) - containing `train` (&, optionally, `validate`)
                key/s + corresponding iterable streams of (x, y) tuples
              n_train (int) - size of training data
              seed (int) - optional random seed to persist random variable
                initialization repeatably across sessions
              verbose (bool) - print to STDOUT to monitor cost/accuracy
              num_cores (int) - number of cores to use for TensorFlow operations
                 (defaults to all cores detected automatically)
              save (bool) - freeze & save trained model binary
              outfile (filepath) - optional path/to/file of model binary
              logging (bool) - save graph_def for visualization with tensorboard
              logdir (filepath) - optional path/to/dir

        Returns: list of cross-validation accuracies (empty if `train` only)
        """
        iters_per_epoch = (n_train // self.hyperparams['n_minibatch']) + \
                          ((n_train % self.hyperparams['n_minibatch']) != 0)

        validate = 'validate' in data_dict.iterkeys()
        cross_vals = []
        best_acc = 0

        tf.set_random_seed(seed)
        config = tf.ConfigProto(inter_op_parallelism_threads = num_cores,
                                intra_op_parallelism_threads = num_cores)

        with tf.Session(config = config) as sesh:
            sesh.run(tf.initialize_all_variables())

            if logging:
                logger = tf.train.SummaryWriter(logdir, sesh.graph_def)

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

                    # validate
                    if validate and i % iters_per_epoch == 0:
                        x, y = data_dict['validate'].next()
                        feed_dict = {self.x: x, self.y: y,
                                    self.dropout: 1.0} # keep prob 1
                        accuracy = sesh.run(self.accuracy, feed_dict)
                        cross_vals.append(accuracy)
                        if accuracy >= best_acc:
                            best_acc = accuracy
                            if save:
                                epoch = (i // iters_per_epoch)
                                print 'Record validation accuracy (epoch {}): '.format(
                                    epoch) + '{}. Freezing!'.format(best_acc)
                                self._freeze(epoch = (i // iters_per_epoch))

                        if verbose:
                            print 'CROSS VAL accuracy at epoch {}: {}'.format(
                                i // iters_per_epoch, accuracy)

            except(KeyboardInterrupt):
                print """
Epochs: {}
Current cross-val accuracies: {}
""".format(i / iters_per_epoch, cross_vals)
                raise

            if save:
                if not validate:
                    self._freeze(epoch = (i // iters_per_epoch))
                with open(outfile, 'wb') as f:
                    f.write(sesh.graph_def.SerializeToString())

            if logging:
                logger.flush()
                logger.close()

        if validate:
            assert len(cross_vals) == self.hyperparams['epochs']
            # latest epoch corresponding to best cross-validation accuracy
            if verbose:
                i, max_ = max(enumerate(cross_vals, 1), key = lambda x: (x[1], x[0]))
                print """
    HYPERPARAMS: {}
    LAYERS:
    {}
    MAX CROSS-VAL ACCURACY (at epoch {}): {}
""".format(self.hyperparams, '\n    '.join(str(l) for l in self.layers), i, max_)

        return cross_vals

    def _freeze(self, epoch):
        """Add nodes to assign weights & biases to constants containing current
        trained values, enabling them to be saved in TensorFlow graph_def """
        regex = re.compile('^[^:]*') # string preceding first `:`
        with tf.name_scope('assign_ops_{}'.format(epoch)):
            for tvar in tf.trainable_variables():
                tf.assign(tvar, tvar.eval(), name =
                          re.match(regex, tvar.name).group(0))

    def predict(self, data, epoch_to_restore = None):
        """Restore Model values for trained variables at given epoch (int), or
        most recent frozen epoch if not specified, and use to generate predictions
        from preprocessed test data.

        Returns: np.array of indices corresponding to categorical predictions
        """
        with tf.Session() as sesh:
            ops = sesh.graph.get_operations()

            # restore weights, biases from last frozen checkpoint
            if not epoch_to_restore:
                match = re.compile('^assign_ops_([^\/]*)').match
                assign_op_epochs = {m.group(1) for m in
                                    (match(op.name) for op in ops) if m}
                try:
                    epoch_to_restore = max(map(int, assign_op_epochs))
                except(ValueError):
                    print 'Designated epoch_to_restore was not frozen.'
                    raise

            KEY = 'assign_ops_{}/'.format(epoch_to_restore)
            assign_ops = [op for op in ops if op.name.startswith(KEY)]
            sesh.run(assign_ops)
            print """Restoring trained values (from epoch {}):
    {}
""".format(epoch_to_restore,
           '\n    '.join(op.name[len(KEY):] for op in assign_ops
                         if not op.name.endswith('value')))

            x, _ = data.next()
            feed_dict = {self.x: x, self.dropout: 1.0}
            predictions = sesh.run(self.predictions, feed_dict)

        return predictions
