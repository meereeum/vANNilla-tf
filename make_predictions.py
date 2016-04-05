import pandas as pd
import tensorflow as tf

from build_model import DataIO, Model, OUTFILES, TARGET_LABEL

########################################################################################

OUTFILES['predictions'] = './predictions.txt'
INFILE = './assignment/test_potus_by_county.csv'
#INFILE = './assignment/train_potus_by_county.csv'


########################################################################################

# ARTIFICIAL NEURAL NET

class Model(Model):
    def predict(self, data, num_cores = None):
        """Restore Model values for trained variables & generate output from test dataset

        Returns: np.array of indices corresponding to categorical predictions
        """
        config = (tf.ConfigProto(inter_op_parallelism_threads = num_cores,
                                intra_op_parallelism_threads = num_cores)
                  if num_cores else None)

        with tf.Session(config = config) as sesh:
            # restore saved weights, biases
            assign_ops = [op for op in tf.Graph.get_operations(sesh.graph)
                          if 'assign_ops' in op.name]
            sesh.run(assign_ops)
            print 'Restored: {}'.format(','.join(op.name for op in assign_ops))
            x, _ = data.next()
            feed_dict = {self.x: x, self.dropout: 1.0}
            predictions = sesh.run(self.predictions, feed_dict)

        return predictions


########################################################################################

def doWork(file_in = INFILE):
    """Resurrect protocols for data preprocessing and artificial neural net model
    construction, and use to generate predictions from input data"""
    df = pd.read_csv(file_in)
    mean = pd.read_csv(OUTFILES['preprocessing_means'])
    std = pd.read_csv(OUTFILES['preprocessing_stddevs'])

    data = DataIO(df, TARGET_LABEL, lambda x: DataIO.gaussianNorm(x, mean, std),
                  [-10, 10]) # TODO: code limits ??

    model = Model(graph_def = OUTFILES['graph_def'])
    predictions = model.predict(data.stream())

    with open(OUTFILES['targets'], 'r') as f:
        targets = f.read().strip().split(',')

    prediction_targets = (targets[i] for i in predictions)

    with open(OUTFILES['predictions'], 'w') as f:
        f.write('\n'.join(prediction_targets))


########################################################################################

if __name__ == '__main__':
    doWork(INFILE)
