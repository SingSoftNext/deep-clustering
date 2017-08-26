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
os.environ["CUDA_VISIBLE_DEVICES"]='1' # restrict processing to a specific GPU

from config import DEEPC_BASE, MODEL_BASE
from visualization import print_examples
from nnet import train_nnet, load_model
from predict import separate_sources

VISUALIZE = False
TEST = True

def main():
    """Main function when called from command line"""

    # Look for a saved model and if found use that
    cached_model_path = os.path.join(DEEPC_BASE, MODEL_BASE + '.h5')
    if os.path.isfile(cached_model_path):
        print('Will use existing model file {}'.format(cached_model_path))
        print("Loading model file...", end='')

    # If no saved model is found, train from scratch
    else:
        print('Beginning training...')
        trn_path = os.path.join(DEEPC_BASE, 'train')
        val_path = os.path.join(DEEPC_BASE, 'valid')
        train_nnet(trn_path, val_path)
    model = load_model(MODEL_BASE)

    # From here on, all the code does is get 2 random speakers from the test
    # set and visualize the outputs and references. You need to have matplotlib
    # installed for this to work.
    if VISUALIZE:
        egs = []
        current_spk = ""
        tst_path = os.path.join(DEEPC_BASE, 'test')
        for line in open(tst_path):
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
        test_in = os.path.join(DEEPC_BASE, 'data/test/mixed.wav')
        test_out = os.path.join(DEEPC_BASE, 'data/test/out')
        separate_sources(test_in, model, 2, test_out)
        print('done')


if __name__ == "__main__":
    main()
