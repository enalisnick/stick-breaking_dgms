import numpy as np
import random
import cPickle as cp
import gzip
import zipfile
import tarfile
import os

import theano
#from sklearn import datasets

def _split_data(data, split):
    starts = np.cumsum(np.r_[0, split[:-1]])
    ends = np.cumsum(split)
    splits = [data[s:e] for s, e in zip(starts, ends)]
    return splits

def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, shared_y

def load_mnist(path, target_as_one_hot=False, flatten=False, split=(50000, 10000, 10000), drop_percentage=0.):
    ''' Loads the MNIST dataset.
    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a ndarray of shape (28*28), i.e. (height*width).
    Example labels are integers between [0,9] respresenting one of the ten classes.
    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (50000, 10000, 10000)
    References
    ----------
    This dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        mnist_picklefile = os.path.join(data_dir, 'mnist.pkl.gz')

        if not os.path.isfile(mnist_picklefile):
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print("Downloading data (16 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, mnist_picklefile)

        # Load the dataset and process it.
        inputs = []
        labels = []
        print("Processing data ...")
        with gzip.open(mnist_picklefile, 'rb') as f:
            trainset, validset, testset = cp.load(f)

        inputs = np.concatenate([trainset[0], validset[0], testset[0]], axis=0).reshape((-1, 1, 28, 28))
        labels = np.concatenate([trainset[1], validset[1], testset[1]], axis=0).astype(np.int8)
        np.savez(path, inputs=inputs, labels=labels)

    print("Loading data ...")
    data = np.load(path)
    inputs, labels = data['inputs'], data['labels']

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    #shuffle                                                                                                                                                                          
    idxs = range(inputs.shape[0])
    random.shuffle(idxs)
    inputs = inputs[idxs,:]
    labels = labels[idxs]

    if target_as_one_hot:
        one_hot_vectors = np.zeros((labels.shape[0], 10), dtype=theano.config.floatX)
        one_hot_vectors[np.arange(labels.shape[0]), labels] = 1
        labels = one_hot_vectors

    datasets_inputs = _split_data(inputs, split)
    datasets_labels = _split_data(labels, split)

    if drop_percentage > 0.:
        N_train = split[0]
        N_wo_label = int(drop_percentage * N_train)
        # split inputs
        labeled_data = datasets_inputs[0][N_wo_label:,:]
        unlabeled_data = datasets_inputs[0][:N_wo_label,:]
        datasets_inputs[0] = labeled_data
        datasets_inputs.insert(2, unlabeled_data)
        # split labels
        labeled_data = datasets_labels[0][N_wo_label:]
        unlabeled_data = datasets_labels[0][:N_wo_label]
        datasets_labels[0] = labeled_data
        datasets_labels.insert(2, unlabeled_data)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets


def load_mnist_w_rotations(path, target_as_one_hot=False, flatten=False, split=(70000, 10000, 20000), drop_percentage=0.):
    ''' Loads the augmented MNIST dataset containing 50k regular MNIST digits and 50k rotated MNIST digits
    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a ndarray of shape (28*28), i.e. (height*width).
    Example labels are integers between [0,9] respresenting one of the ten classes.
    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (70000, 10000, 20000)
    References
    ----------
    The regular MNIST portion of this dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    The rotated MNIST portion comes from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        mnist_picklefile = os.path.join(data_dir, 'mnist_plus_rot.pkl.gz')

        if not os.path.isfile(mnist_picklefile):
            import urllib
            origin = 'http://www.ics.uci.edu/~enalisni/mnist_plus_rot.pkl.gz'
            print("Downloading data (100 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, mnist_picklefile)

        with gzip.open(mnist_picklefile, 'rb') as f:
            data = cp.load(f)
        cp.dump(data, open(os.path.join(data_dir, 'mnist_plus_rot.pkl'), 'wb'), protocol=cp.HIGHEST_PROTOCOL)

    else:
        data = np.load(path)

    inputs, labels = data['inputs'], data['labels']

    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    #shuffle                                                                                                                                                                                                            
    idxs = range(inputs.shape[0])
    random.shuffle(idxs)
    inputs = inputs[idxs,:]
    labels = labels[idxs]

    if target_as_one_hot:
        one_hot_vectors = np.zeros((labels.shape[0], 10), dtype=theano.config.floatX)
        one_hot_vectors[np.arange(labels.shape[0]), labels.astype(int)] = 1
        labels = one_hot_vectors

    datasets_inputs = _split_data(inputs, split)
    datasets_labels = _split_data(labels, split)

    if drop_percentage > 0.:
        N_train = split[0]
        N_wo_label = int(drop_percentage * N_train)
        # split inputs                                                                                                    
        labeled_data = datasets_inputs[0][N_wo_label:,:]
        unlabeled_data = datasets_inputs[0][:N_wo_label,:]
        datasets_inputs[0] = labeled_data
        datasets_inputs.insert(2, unlabeled_data)
        # split labels                                              
        labeled_data = datasets_labels[0][N_wo_label:]
        unlabeled_data = datasets_labels[0][:N_wo_label]
        datasets_labels[0] = labeled_data
        datasets_labels.insert(2, unlabeled_data)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets

def load_svhn_pca(path, target_as_one_hot=True, train_valid_split=(65000, 8254), drop_percentage=0.):
    ''' Loads the Street View House Numbers (SVHN) dataset pre-processed with PCA, reduced to 500 dimensions.
        Example labels are integers between [0,9] respresenting one of the ten classes.
        Parameters
        ----------
        path : str
        The path to the dataset file (.pkl).
        target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
        flatten : {True, False}, optional
        If True, represents each individual example as a vector.
        split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (65000, 8254)
        References
        ----------
        The original dataset can be attained at http://ufldl.stanford.edu/housenumbers/
        '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        svhn_picklefile = os.path.join(data_dir, 'svhn_pca.pkl.gz')
        
        if not os.path.isfile(svhn_picklefile):
            import urllib
            origin = 'http://www.ics.uci.edu/~enalisni/svhn_pca.pkl.gz'
            print("Downloading data (370 Mb) from {} ...".format(origin))
            urllib.urlretrieve(origin, svhn_picklefile)
        
        with gzip.open(svhn_picklefile, 'rb') as f:
            data = cp.load(f)
        cp.dump(data, open(os.path.join(data_dir, 'svhn_pca.pkl'), 'wb'), protocol=cp.HIGHEST_PROTOCOL)
    
    else:
        data = cp.load(open(path,'rb'))

    train_inputs = data['train_data']
    test_inputs = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    
    #shuffle
    idxs = range(train_inputs.shape[0])
    random.shuffle(idxs)
    train_inputs = train_inputs[idxs,:]
    train_labels = train_labels[idxs]
    
    if target_as_one_hot:
        one_hot_vectors_train = np.zeros((train_labels.shape[0], 10), dtype=theano.config.floatX)
        for idx in xrange(train_labels.shape[0]):
            one_hot_vectors_train[idx, train_labels[idx]] = 1.
        train_labels = one_hot_vectors_train

        one_hot_vectors_test = np.zeros((test_labels.shape[0], 10), dtype=theano.config.floatX)
        for idx in xrange(test_labels.shape[0]):
            one_hot_vectors_test[idx, test_labels[idx]] = 1.
        test_labels = one_hot_vectors_test

    datasets_inputs = [ train_inputs[:train_valid_split[0],:], train_inputs[-1*train_valid_split[1]:,:], test_inputs ]
    datasets_labels = [ train_labels[:train_valid_split[0]], train_labels[-1*train_valid_split[1]:], test_labels ]

    if drop_percentage > 0.:
        N_train = train_valid_split[0]
        N_wo_label = int(drop_percentage * N_train)
        # split inputs
        labeled_input_data = datasets_inputs[0][N_wo_label:,:]
        unlabeled_input_data = datasets_inputs[0][:N_wo_label,:]
        datasets_inputs[0] = labeled_input_data
        datasets_inputs.insert(2, unlabeled_input_data)
        # split labels
        labeled_label_data = datasets_labels[0][N_wo_label:]
        unlabeled_label_data = datasets_labels[0][:N_wo_label]
        datasets_labels[0] = labeled_label_data
        datasets_labels.insert(2, unlabeled_label_data)

    datasets = [_shared_dataset((i, l)) for i, l in zip(datasets_inputs, datasets_labels)]
    return datasets

