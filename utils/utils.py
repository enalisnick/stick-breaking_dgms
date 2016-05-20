import os
import numpy as np
import theano
import theano.tensor as T

### Misc ###

def sharedX(value, name=None, borrow=True, keep_on_cpu=False):
    """ Transform value into a shared variable of type floatX """
    if keep_on_cpu:
        return T._shared(theano._asarray(value, dtype=theano.config.floatX),name=name, borrow=borrow)
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX), name=name, borrow=borrow)

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass
    return path

### Weights initializers ###

def guess_init_scale(shape):
    """ Provides appropriate scale for initialization of the weights. """
    if len(shape) == 2:
        # For feedforward networks (see http://deeplearning.net/tutorial/mlp.html#going-from-logistic-regression-to-mlp)
        return np.sqrt(6. / (shape[0] + shape[1]))
    elif len(shape) == 4:
        # For convnet (see http://deeplearning.net/tutorial/lenet.html)
        fan_in = np.prod(shape[1:])
        fan_out = shape[0] * np.prod(shape[2:])
        return np.sqrt(6. / (fan_in + fan_out))
    else:
        raise ValueError("Don't know what to do in this case!")


def init_params_zeros(shape=None, values=None, name=None):
    if values is None:
        values = np.zeros(shape, dtype=theano.config.floatX)

    return theano.shared(value=values, name=name)


def init_params_randn(rng, shape=None, sigma=0.01, values=None, name=None):
    if sigma is None:
        sigma = guess_init_scale(shape)

    if values is None:
        values = sigma * rng.randn(*shape)

    return sharedX(values, name=name)


def init_params_uniform(rng, shape=None, scale=None, values=None, name=None):
    if scale is None:
        scale = guess_init_scale(shape)

    if values is None:
        values = rng.uniform(-scale, scale, shape)

    return sharedX(values, name=name)


def init_params_orthogonal(rng, shape=None, values=None, name=None):
    if values is None:
        # initialize w/ orthogonal matrix.  code taken from:
        # https://github.com/mila-udem/blocks/blob/master/blocks/initialization.py
        M = np.asarray(rng.standard_normal(size=shape), dtype=theano.config.floatX)
        Q, R = np.linalg.qr(M)
        values = Q * np.sign(np.diag(R)) * 0.01

    return sharedX(values, name=name)
