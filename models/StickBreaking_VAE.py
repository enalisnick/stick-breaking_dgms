import numpy as np
import theano
import theano.tensor as T

from variational_coders.encoders import StickBreakingEncoder
from variational_coders.decoders import Decoder

### Stick-Breaking VAE ###
class StickBreaking_VAE(object):
    def __init__(self, rng, input, batch_size, layer_sizes, 
                 layer_types, activations, latent_size, out_activation): # architecture specs
        
        # check lists are correct sizes
        assert len(layer_types) == len(layer_sizes) - 1
        assert len(activations) == len(layer_sizes) - 1
    
        # Set up the NN that parametrizes the encoder
        layer_specs = zip(layer_types, layer_sizes, layer_sizes[1:])
        self.encoding_layers = []
        next_layer_input = input
        activation_counter = 0        
        for layer_type, n_in, n_out in layer_specs:
            next_layer = layer_type(rng=rng, input=next_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out)
            next_layer_input = next_layer.output
            self.encoding_layers.append(next_layer)
            activation_counter += 1

        # init encoder
        self.encoder = StickBreakingEncoder(rng, input=next_layer_input, batch_size=batch_size, in_size=layer_sizes[-1], latent_size=latent_size)
        
        # init decoder 
        self.decoder = Decoder(rng, input=self.encoder.latent_vars, latent_size=latent_size, out_size=layer_sizes[-1], activation=activations[-1])

        # setup the NN that parametrizes the decoder (generative model)
        layer_specs = zip(reversed(layer_types), reversed(layer_sizes), reversed(layer_sizes[:-1]))
        self.decoding_layers = []
        # add output activation as first activation.  last act. taken care of by the decoder
        activations = [out_activation] + activations[:-1]
        activation_counter = len(activations)-1
        next_layer_input = self.decoder.output
        for layer_type, n_in, n_out in layer_specs:
            # supervised decoding layers
            next_layer = layer_type(rng=rng, input=next_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out)
            next_layer_input = next_layer.output
            self.decoding_layers.append(next_layer)
            activation_counter -= 1
            
        # Grab all the parameters--only need to get one half since params are tied
        self.params = [p for layer in self.encoding_layers for p in layer.params] + self.encoder.params + self.decoder.params + [p for layer in self.decoding_layers for p in layer.params]

        # Grab the posterior params
        self.post_a = self.encoder.a
        self.post_b = self.encoder.b

        # grab the kl-divergence functions
        self.calc_kl_divergence = self.encoder.calc_kl_divergence

        # Grab the reconstructions and predictions
        self.x_recon = next_layer_input
