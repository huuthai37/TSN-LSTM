import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--process', help='Process', default='train')
parser.add_argument('-data', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-b', '--batch', help='Batch size', default=16, type=int)
parser.add_argument('-c', '--classes', help='Number of classes', default=101, type=int)
parser.add_argument('-e', '--epoch', help='Number of epochs', default=5, type=int)
parser.add_argument('-dropout', '--dropout', help='Dropout', default=0.8, type=float)
parser.add_argument('-r', '--retrain', help='Number of old epochs when retrain', default=0, type=int)
parser.add_argument('-cross', '--cross', help='Cross fold', default=1, type=int)
parser.add_argument('-s', '--summary', help='Show model', default=0, type=int)
parser.add_argument('-lr', '--lr', help='Learning rate', default=5e-3, type=float)
parser.add_argument('-decay', '--decay', help='Decay', default=1e-6, type=float)
parser.add_argument('-fine', '--fine', help='Fine-tuning', default=1, type=int)
parser.add_argument('-n', '--neural', help='LSTM neural', default=256, type=int)
parser.add_argument('-t', '--temporal', help='Temporal rate', default=1, type=int)
args = parser.parse_args()
print args

import sys
import config
import models
from keras import optimizers

process = args.process
if process == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif process == 'retrain':
    train = True
    retrain = True
    old_epochs = args.retrain
else:
    train = False
    retrain = False
    old_epochs = 0

batch_size = args.batch
classes = args.classes
epochs = args.epoch
cross_index = args.cross
dataset = args.dataset
temp_rate = args.temporal

seq_len = 3
n_neurons = args.neural
dropout = args.dropout
pre_file = 'incept229_temporal{}_lstm{}'.format(temp_rate,n_neurons)

if train & (not retrain):
    weights = 'imagenet'
else:
    weights = None
if args.fine == 1:
    fine = True
else:
    fine = False

result_model = models.InceptionTemporalLSTMConsensus(
                    n_neurons=n_neurons, seq_len=seq_len, classes=classes, 
                    weights=weights, dropout=dropout, fine=fine, retrain=retrain,
                    pre_file=pre_file,old_epochs=old_epochs,cross_index=cross_index)


if (args.summary == 1):
    result_model.summary()
    sys.exit()

lr = args.lr 
decay = args.decay

result_model.compile(loss='categorical_crossentropy',
                     optimizer=optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False),
                     metrics=['accuracy'])
    

if train:
    models.train_process(result_model, pre_file, data_type=[temp_rate], epochs=epochs, dataset=dataset,
        retrain=retrain,  classes=classes, cross_index=cross_index, 
        seq_len=seq_len, old_epochs=old_epochs, batch_size=batch_size,fine=fine)

else:
    models.test_process(result_model, pre_file, data_type=[temp_rate], epochs=epochs, dataset=dataset,
        classes=classes, cross_index=cross_index,
        seq_len=seq_len, batch_size=batch_size)
    