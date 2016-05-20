import numpy as np
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):
        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
        self.W = W
        self.b = b

        self.output = self.activation( T.dot(self.input, self.W) + self.b )
    
        # parameters of the model
        self.params = [self.W, self.b]

        
class ResidualHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):
        
        super(ResidualHiddenLayer, self).__init__(rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, activation=activation)
        
        # F(h_l-1) + h_l-1
        self.output += self.input


