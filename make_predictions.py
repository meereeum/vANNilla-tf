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
    def predict(self, data):
        """Restore Model values for trained variables & generate output from test dataset

        Returns: np.array of indices corresponding to categorical predictions
        """
        with tf.Session() as sesh:
            # restore saved weights, biases
            #assign_ops = sesh.graph.get_collection('assign_ops')
            #sesh.run(sesh.graph.get_collection('assign_ops'))
            KEY = 'assign_ops/'
            assign_ops = [op for op in sesh.graph.get_operations()
                          if op.name.startswith(KEY)]
            sesh.run(assign_ops)
            print """Restored (trained) values:
    {}
""".format('\n    '.join(op.name[len(KEY):] for op in assign_ops
                         if not op.name.endswith('value')))

            x, _ = data.next()
            import code; code.interact(local=locals())
            print x
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
        f.write('\n'.join(prediction_targets) + '\r')

    print "Predictions saved to: {}".format(OUTFILES['predictions'])


########################################################################################

if __name__ == '__main__':
    doWork(INFILE)
