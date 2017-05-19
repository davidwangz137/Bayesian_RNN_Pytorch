'''
Dataset objects
    Data
    Method for getting batch
    Method for checking if end of batch
'''

# External libraries
import os
import cPickle as pickle

# Internal libraries
import args


class Dataset(object):

    def __init__(self, args):
        raise NotImplementedError


class PTBDataset(Dataset):

    def __init__(self, args):
        REMOVEME = 0

    def epoch_ended(self):
        return REMOVEME >= 10

    def get_batch(self, split='train'):
        REMOVEME += 1
        return REMOVEME

    def reset_epoch(self):
        REMOVEME = 0


if __name__ == "__main__":
    args = args.get_args()
    dataset = PTBDataset(args)
    pickle.dump(args.dataset_file, open(args.dataset_file, 'wb'))
