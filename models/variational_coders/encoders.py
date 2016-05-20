import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from models.neural_net.activation_fns import Softplus, Beta_fn

### GAUSSIAN ###

# Regular Encoder for Gaussian Latent Variables 
class GaussEncoder(object):
    def __init__(self, rng, input, batch_size, in_size, latent_size, W_mu = None, W_sigma = None):
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.input = input
        
        # setup variational params
        if W_mu is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, latent_size)), dtype=theano.config.floatX)
            W_mu = theano.shared(value=W_values, name='W_mu')
        if W_sigma is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, latent_size)), dtype=theano.config.floatX)
            W_sigma = theano.shared(value=W_values, name='W_sigma')
        self.W_mu = W_mu
        self.W_sigma = W_sigma

        # compute Normal samples
        std_norm_samples = T.cast(self.srng.normal(size=(batch_size, latent_size), avg=0.0, std=1.0), dtype=theano.config.floatX)
        self.mu = T.dot(self.input, self.W_mu)
        self.log_sigma = T.dot(self.input, self.W_sigma)
        self.latent_vars = self.mu + T.exp(self.log_sigma) * std_norm_samples
        
        self.params = [self.W_mu, self.W_sigma]

    # Assumes diagonal Gaussians                                                                                              
    def calc_kl_divergence(self, prior_mu, prior_sigma):
        kl = -T.log(prior_sigma**2)
        kl += -(T.exp(2*self.log_sigma) + (self.mu - prior_mu)**2)/(prior_sigma**2)
        kl += 2*self.log_sigma + 1.
        return -0.5*kl.sum(axis=1)


# Encoder for Gaussian and Label Latent Variables
class Gauss_Encoder_w_Labels(GaussEncoder):
    def __init__(self, rng, input, batch_size, in_size, label_size, latent_size, label_fn, W_y = None, b_y = None, W_mu = None, W_sigma = None):
        self.label_fn = label_fn

        # init parent class
        super(Gauss_Encoder_w_Labels, self).__init__(rng=rng, input=input, batch_size=batch_size, in_size=in_size, latent_size=latent_size, W_mu=W_mu, W_sigma=W_sigma)

        # setup label prediction params                                                                                                                 
        if W_y is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, label_size)), dtype=theano.config.floatX)
            W_y = theano.shared(value=W_values, name='W_y')
        if b_y is None:
            b_values = np.zeros((label_size,), dtype=theano.config.floatX)
            b_y = theano.shared(value=b_values, name='b_y')
        self.W_y = W_y
        self.b_y = b_y

        # compute the label probabilities                                                                                           
        self.y_probs = self.label_fn(T.dot(self.input, self.W_y) + self.b_y)
        self.y_probs = T.maximum(T.minimum(self.y_probs, 1-1e-4), 1e-4)  # Force 0 < output < 1                                                                                                                                  
        self.params += [self.W_y, self.b_y]


### STICK BREAKING ###

# Regular Encoder for Stick-Breaking Latent Variables 
class StickBreakingEncoder(object):
    def __init__(self, rng, input, batch_size, in_size, latent_size, W_a = None, W_b = None, epsilon = 0.01):
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.input = input
        
        # setup variational params
        if W_a is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, latent_size-1)), dtype=theano.config.floatX)
            W_a = theano.shared(value=W_values, name='W_a')
        if W_b is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, latent_size-1)), dtype=theano.config.floatX)
            W_b = theano.shared(value=W_values, name='W_b')
        self.W_a = W_a
        self.W_b = W_b

        # compute Kumaraswamy samples                                                                                                                                                      
        uniform_samples = T.cast(self.srng.uniform(size=(batch_size, latent_size-1), low=0.01, high=0.99), theano.config.floatX)
        self.a = Softplus(T.dot(self.input, self.W_a))
        self.b = Softplus(T.dot(self.input, self.W_b))
        v_samples = (1-(uniform_samples**(1/self.b)))**(1/self.a)

        # setup variables for recursion                                                                                                                                   
        stick_segment = theano.shared(value=np.zeros((batch_size,), dtype=theano.config.floatX), name='stick_segment')
        remaining_stick = theano.shared(value=np.ones((batch_size,), dtype=theano.config.floatX), name='remaining_stick')

        def compute_latent_vars(i, stick_segment, remaining_stick, v_samples):
            # compute stick segment                                                                                                     
            stick_segment = v_samples[:,i] * remaining_stick
            remaining_stick *= (1-v_samples[:,i])
            return (stick_segment, remaining_stick)

        (stick_segments, remaining_sticks), updates = theano.scan(fn=compute_latent_vars,
                                                                  outputs_info=[stick_segment, remaining_stick],sequences=T.arange(latent_size-1),
                                                                  non_sequences=[v_samples], strict=True)

        self.avg_used_dims = T.mean(T.sum(remaining_sticks > epsilon, axis=0))
        self.latent_vars = T.transpose(T.concatenate([stick_segments, T.shape_padaxis(remaining_sticks[-1, :],axis=1).T], axis=0))
        
        self.params = [self.W_a, self.W_b]
                                                                                              
    def calc_kl_divergence(self, prior_alpha, prior_beta):
        # compute taylor expansion for E[log (1-v)] term                                                                                                                                             
        # hard-code so we don't have to use Scan()                                                                                                                                                   
        kl = 1./(1+self.a*self.b) * Beta_fn(1./self.a, self.b)
        kl += 1./(2+self.a*self.b) * Beta_fn(2./self.a, self.b)
        kl += 1./(3+self.a*self.b) * Beta_fn(3./self.a, self.b)
        kl += 1./(4+self.a*self.b) * Beta_fn(4./self.a, self.b)
        kl += 1./(5+self.a*self.b) * Beta_fn(5./self.a, self.b)
        kl += 1./(6+self.a*self.b) * Beta_fn(6./self.a, self.b)
        kl += 1./(7+self.a*self.b) * Beta_fn(7./self.a, self.b)
        kl += 1./(8+self.a*self.b) * Beta_fn(8./self.a, self.b)
        kl += 1./(9+self.a*self.b) * Beta_fn(9./self.a, self.b)
        kl += 1./(10+self.a*self.b) * Beta_fn(10./self.a, self.b)
        kl *= (prior_beta-1)*self.b

        # use another taylor approx for Digamma function                                                                                                                                             
        psi_b_taylor_approx = T.log(self.b) - 1./(2 * self.b) - 1./(12 * self.b**2)
        kl += (self.a-prior_alpha)/self.a * (-0.57721 - psi_b_taylor_approx - 1/self.b) #T.psi(self.posterior_b)                                                                                        

        # add normalization constants                                                                                                                                                                
        kl += T.log(self.a*self.b) + T.log(Beta_fn(prior_alpha, prior_beta))

        # final term                                                                                                                                                                                 
        kl += -(self.b-1)/self.b

        return kl.sum(axis=1)



# Encoder for StickBreaking and Label Latent Variables
class StickBreaking_Encoder_w_Labels(StickBreakingEncoder):
    def __init__(self, rng, input, batch_size, in_size, label_size, latent_size, label_fn, W_y = None, b_y = None, W_a = None, W_b = None):
        self.label_fn = label_fn

        # init parent class
        super(StickBreaking_Encoder_w_Labels, self).__init__(rng=rng, input=input, batch_size=batch_size, in_size=in_size, latent_size=latent_size, W_a=W_a, W_b=W_b)

        # setup label prediction params                                                                                                                 
        if W_y is None:
            W_values = np.asarray(0.01 * rng.standard_normal(size=(in_size, label_size)), dtype=theano.config.floatX)
            W_y = theano.shared(value=W_values, name='W_y')
        if b_y is None:
            b_values = np.zeros((label_size,), dtype=theano.config.floatX)
            b_y = theano.shared(value=b_values, name='b_y')
        self.W_y = W_y
        self.b_y = b_y

        # compute the label probabilities                                                                                           
        self.y_probs = self.label_fn(T.dot(self.input, self.W_y) + self.b_y)
        self.y_probs = T.maximum(T.minimum(self.y_probs, 1-1e-4), 1e-4)  # Force 0 < output < 1                                                                                                                                  
        self.params += [self.W_y, self.b_y]
