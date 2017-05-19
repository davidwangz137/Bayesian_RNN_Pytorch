'''
Runs the pipeline
'''

# External libraries
from __future__ import print_function
import os
import time
import numpy
import cPickle as pickle

# Internal libraries
import args
reload(args)
import dataset
import rnn

'''
Arguments
'''

args = args.get_args()


'''
Create/Load model
'''

model = rnn.get_model(args)


'''
Load dataset
'''

if not os.path.isfile(args.dataset_file):
    assert False, "No dataset file to load from!"

dataset = pickle.load(open(args.dataset_file, 'rb'))

'''
Train and save model
'''

for i in range(args.num_epochs):

    # Train model for an epoch
    while not dataset.epoch_ended():
        batch = dataset.get_batch(split='train')
        model.train(batch)

    # Reset the dataset for a new epoch
    dataset.reset_epoch()

    # Save model

    # Evaluate on the validation or test set
    while not dataset.epoch_ended():
        batch = dataset.get_batch(split='validation')
        model_res = model.evaluate(batch)
        dataset.performance(model_res)
    dataset.print_evaluation()


'''
Test model
'''

while not dataset.epoch_ended():
    batch = dataset.get_batch(split='validation')
    model_res = model.evaluate(batch)
    dataset.performance(model_res)
dataset.print_evaluation()


