import pandas as pd
import tensorflow as tf

from build_model import DataIO, Model, OUTFILES, TARGET_LABEL

OUTFILES['predictions'] = './predictions.txt'
INFILE = './assignment/test_potus_by_county.csv'
#INFILE = './assignment/train_potus_by_county.csv'

class Model(Model):
    def predict(self, data):
        with tf.Session() as sesh:
            # restore saved weights, biases
            assign_ops = [op for op in tf.Graph.get_operations(sesh.graph)
                          if 'assign_ops' in op.name]
            sesh.run(assign_ops)
            x, _ = data.next()
            feed_dict = {self.x: x, self.dropout: 1.0}
            predictions = sesh.run(self.predictions, feed_dict)
        return predictions

def doWork(file_in):
    df = pd.read_csv(file_in)
    mean = pd.read_csv(OUTFILES['preprocessing_means'])
    std = pd.read_csv(OUTFILES['preprocessing_stddevs'])

    with open(OUTFILES['targets'], 'r') as f:
        targets = f.read().strip().split(',')

    data = DataIO(df, TARGET_LABEL, lambda x: DataIO.gaussianNorm(x, mean, std),
                  [-10, 10]) # TODO: code limits ??

    model = Model(graph_def = OUTFILES['graph_def'])
    predictions = model.predict(data.stream())
    prediction_targets = (targets[i] for i in predictions)

    with open(OUTFILES['predictions'], 'w') as f:
        f.write('\n'.join(prediction_targets))


if __name__ == '__main__':
    doWork(INFILE)
