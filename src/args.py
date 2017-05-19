'''
Arguments of the pipeline
'''

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RNN run pipeline script parser')

    # Dataset
    parser.add_argument('--data', type=str, default='./data/penn', help='location of the data corpus')
    parser.add_argument('-dataset_file', type=str, help='location of the dataset object .pkl', required=True)

    # Training parameters
    parser.add_argument('-num_epochs', type=int, help='number of epoches to train for', required=True)

    # Model parameters
    return parser.parse_args()
