# Keras implementation of [Deep Clustering paper](https://arxiv.org/abs/1508.04306)

This is a keras implementation of the Deep Clustering algorithm described at https://arxiv.org/abs/1508.04306. It is not yet finished. Most of this code was implemented by [Valter Akira Miasato Filho](https://github.com/akira-miasato). 

Requirements
------------

1. System library: 
  * libsndfile1 (installed via apt-get on Ubuntu 16.04)

2. Python packages (I used Anaconda and Python 3.5):
  * Theano (pip install git+git://github.com/Theano/Theano.git)
  * keras (pip install keras)
  * pysoundfile (pip install pysoundfile)
  * numpy (conda install numpy)
  * scikit-learn (conda install scikit-learn)
  * matplotlib (conda install matplotlib) (only used for visualization)


Training the network
--------------------

First, create a directory to hold the training configuration files. Look in `config.py` for `DEEPC_BASE` for the default value. 

There must be two text files in that directory: `train` and `valid`. They must contain the paths to your training and validation data. The format of the files are
```
path/to/audioFile1 spkr1
path/to/audioFile2 spkr2
path/to/audioFile3 spkr1
```
where `spkr1`, `spkr2` identify the speaker that uttered the recorded sentence. 


The current implementation should work with any sample rate, but experiments were conducted only with 8kHz audio. It was already tested with flac and wav files, but it should work with all formats supported by [pysoundfile/libsndfile](http://www.mega-nerd.com/libsndfile/#Features).


After creating `train` and `valid`, you may start training the network with the command:
```
python main.py
```
Please check the `main.py` script if you wish to use other features from this project, such
as output visualization. By default, the script will do a prediction using as input `DEEPC_BASE/data/test/mixed.wav` and creating outputs `out_n.wav` in the same directory, where `n` is the number of the speaker.


As of February, 2017, the original authors of this code have stopped working on the project. However, the SingSoftNext team is continuing development.


References
----------
* https://arxiv.org/abs/1508.04306
