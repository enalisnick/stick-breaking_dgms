import numpy as np
import theano
import theano.tensor as T

### Regular Decoder
class Decoder(object):
    def __init__(self, rng, input, latent_size, out_size, activation, W_z = None, b = None):
        self.input = input
        self.activation = activation

        # setup the params                                                                                                                          
        if W_z is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(latent_size, out_size)), dtype=theano.config.floatX)
            W_z = theano.shared(value=W_values, name='W_hid_z')
        if b is None:
            b_values = np.zeros((out_size,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
        self.W_z = W_z
        self.b = b
        
        self.pre_act_out = T.dot(self.input, self.W_z) + self.b
        self.output = self.activation(self.pre_act_out)
        
        # gather parameters
        self.params = [self.W_z, self.b]

### Supervised Decoder
class Supervised_Decoder(Decoder):
    def __init__(self, rng, input, labels, latent_size, label_size, out_size, activation, W_z = None, W_y = None, b = None):
        self.labels = labels

        # init parent class                                                     
        super(Supervised_Decoder, self).__init__(rng=rng, input=input, latent_size=latent_size, out_size=out_size, activation=activation, W_z=W_z, b=b)

        # setup the params                                                                                                                                         
        if W_y is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(label_size, out_size)), dtype=theano.config.floatX)
            W_y = theano.shared(value=W_values, name='W_y')
        self.W_y = W_y

        self.output = self.activation( self.pre_act_out + T.dot(self.labels, self.W_y) )

        # gather parameters                     
        self.params += [self.W_y]
    
### Marginalized Decoder (for semi-supervised model)
class Marginalized_Decoder(Decoder):
    def __init__(self, rng, input, batch_size, latent_size, label_size, out_size, activation, W_z, W_y, b):
        
        # init parent class                          
        super(Marginalized_Decoder, self).__init__(rng=rng, input=input, latent_size=latent_size, out_size=out_size, activation=activation, W_z=W_z, b=b)

        # setup the params           
        self.W_y = W_y

        # compute marginalized outputs                                                                                                                 
        labels_tensor = T.extra_ops.repeat( T.shape_padaxis(T.eye(n=label_size, m=label_size), axis=0), repeats=batch_size, axis=0)
        self.output = self.activation(T.extra_ops.repeat(T.shape_padaxis(T.dot(self.input, self.W_z), axis=1), repeats=label_size, axis=1) + T.dot(labels_tensor, self.W_y) + self.b)
        
        # no params here since we'll grab them from the supervised decoder

