# -*- coding: utf-8 -*-
# pylint: disable=C0103,R0912,R0913,R0914,R0915
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf

Main code with examples for the most important function calls. None of this
will work if you haven't prepared your train/valid/test file lists.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # filter all except errors

from visualization import print_examples
from nnet import train_nnet, load_model
from predict import separate_sources

VISUALIZE = False
TEST = True

def main():
    """Main function when called from command line"""
    if os.path.isfile('model.h5'):
        print('Will use existing model file')
    else:
        print('Beginning training...')
        train_nnet('train', 'valid')
    print("Loading model file...", end='')
    model = load_model('model')
    print('done')

    # From here on, all the code does is get 2 random speakers from the test
    # set and visualize the outputs and references. You need to have matplotlib
    # installed for this to work.
    if VISUALIZE:
        egs = []
        current_spk = ""
        for line in open('test'):
            line = line.strip().split()
            if len(line) != 2:
                continue
            w, s = line
            if s != current_spk:
                egs.append(w)
                current_spk = s
                if len(egs) == 2:
                    break
        print_examples(egs, model, db_threshold=40, ignore_background=True)

    # If you wish to test source separation, generate a mixed 'mixed.wav'
    # file and test with the following line
    if TEST:
        print('Beginning test...', end='')
        separate_sources('data/test/mixed.wav', model, 2, 'data/test/out')
        print('done')


if __name__ == "__main__":
    main()
