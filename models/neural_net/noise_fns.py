import numpy as np
import theano
import theano.tensor as T

def no_noise(input):
    # needed because dnn pseudo ensemble code assumes each input / hidden layer gets noise
    return input

def dropout_noise(rng, input, p=0.5):
    
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    # Bernoulli(1-p) multiplicative noise                                                 
    mask = T.cast(srng.binomial(n=1, p=1-p, size=input.shape), theano.config.floatX)
    return mask * input


def beta_noise(rng, input):

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    
    # Beta(.5,.5) multiplicative noise                       
    mask = T.cast(T.sin( (np.pi / 2.0) * srng.uniform(size=input.shape, low=0.0, high=1.0) )**2, theano.config.floatX)
    return mask * input

def poisson_noise(rng, input, lam=0.5):

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    # Poisson noise
    mask = T.cast(srng.poisson(lam=lam, size=input.shape), theano.config.floatX)
    return mask * input
