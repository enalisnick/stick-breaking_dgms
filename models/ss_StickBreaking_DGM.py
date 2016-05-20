import numpy as np
import theano
import theano.tensor as T

from variational_coders.encoders import StickBreaking_Encoder_w_Labels
from variational_coders.decoders import Supervised_Decoder, Marginalized_Decoder

### Gaussian Semi-Supervised DGM ###
class SS_StickBreaking_DGM(object):
    def __init__(self, rng, sup_input, un_sup_input, labels, 
                 sup_batch_size, un_sup_batch_size, 
                 layer_sizes, layer_types, activations,
                 label_size, latent_size, out_activation, label_fn): # architecture specs
        
        # check lists are correct sizes
        assert len(layer_types) == len(layer_sizes) - 1
        assert len(activations) == len(layer_sizes) - 1
        assert label_size > 1 # labels need to be one-hot encoded!
    
        # Set up the NN that parametrizes the encoder
        layer_specs = zip(layer_types, layer_sizes, layer_sizes[1:])
        self.encoding_layers = []
        next_sup_layer_input = sup_input
        next_un_sup_layer_input = un_sup_input
        activation_counter = 0        
        for layer_type, n_in, n_out in layer_specs:
            next_sup_layer = layer_type(rng=rng, input=next_sup_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out)
            next_sup_layer_input = next_sup_layer.output
            self.encoding_layers.append(next_sup_layer)
            next_un_sup_layer = layer_type(rng=rng, input=next_un_sup_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out, W=next_sup_layer.W, b=next_sup_layer.b)
            next_un_sup_layer_input = next_un_sup_layer.output
            activation_counter += 1

        # init encoders -- one supervised, one un_supervised
        self.supervised_encoder = StickBreaking_Encoder_w_Labels(rng, input=next_sup_layer_input, batch_size=sup_batch_size, 
                                                                 in_size=layer_sizes[-1], label_size=label_size, latent_size=latent_size, label_fn=label_fn)
        self.un_supervised_encoder = StickBreaking_Encoder_w_Labels(rng, input=next_un_sup_layer_input, batch_size=un_sup_batch_size, 
                                                                    in_size=layer_sizes[-1], label_size=label_size, latent_size=latent_size, label_fn=label_fn, 
                                                                    W_y = self.supervised_encoder.W_y, b_y = self.supervised_encoder.b_y, 
                                                                    W_a = self.supervised_encoder.W_a, W_b = self.supervised_encoder.W_b)

        # init decoders -- one supervised, one up_supervised
        self.supervised_decoder = Supervised_Decoder(rng, input=self.supervised_encoder.latent_vars, labels=labels, latent_size=latent_size, 
                                                     label_size=label_size, out_size=layer_sizes[-1], activation=activations[-1])
        self.un_supervised_decoder = Marginalized_Decoder(rng, input=self.un_supervised_encoder.latent_vars, batch_size=un_sup_batch_size, latent_size=latent_size, label_size=label_size, 
                                                     out_size=layer_sizes[-1], activation=activations[-1], 
                                                     W_z=self.supervised_decoder.W_z, W_y=self.supervised_decoder.W_y, b=self.supervised_decoder.b)

        # setup the NN that parametrizes the decoder (generative model)
        layer_specs = zip(reversed(layer_types), reversed(layer_sizes), reversed(layer_sizes[:-1]))
        self.decoding_layers = []
        # add output activation as first activation.  last act. taken care of by the decoder
        activations = [out_activation] + activations[:-1]
        activation_counter = len(activations)-1
        next_sup_layer_input = self.supervised_decoder.output
        next_un_sup_layer_input = self.un_supervised_decoder.output
        for layer_type, n_in, n_out in layer_specs:
            # supervised decoding layers
            next_sup_layer = layer_type(rng=rng, input=next_sup_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out)
            next_sup_layer_input = next_sup_layer.output
            self.decoding_layers.append(next_sup_layer)
            # un supervised decoding layers
            next_un_sup_layer = layer_type(rng=rng, input=next_un_sup_layer_input, activation=activations[activation_counter], n_in=n_in, n_out=n_out, W=next_sup_layer.W, b=next_sup_layer.b)
            next_un_sup_layer_input = next_un_sup_layer.output
            activation_counter -= 1
            
        # Grab all the parameters--only need to get one half since params are tied
        self.params = [p for layer in self.encoding_layers for p in layer.params] + self.supervised_encoder.params + self.supervised_decoder.params + [p for layer in self.decoding_layers for p in layer.params]

        # Grab the posterior params
        self.sup_post_a = self.supervised_encoder.a
        self.sup_post_b = self.supervised_encoder.b
        self.un_sup_post_a = self.un_supervised_encoder.a
        self.un_sup_post_b = self.un_supervised_encoder.b

        # grab the kl-divergence functions
        self.calc_sup_kl_divergence = self.supervised_encoder.calc_kl_divergence
        self.calc_un_sup_kl_divergence = self.un_supervised_encoder.calc_kl_divergence

        # Grab the reconstructions and predictions
        self.x_recon_sup = next_sup_layer_input
        self.x_recon_un_sup = next_un_sup_layer_input
        self.y_probs_sup = self.supervised_encoder.y_probs
        self.y_probs_un_sup = self.un_supervised_encoder.y_probs
        self.y_preds_sup = T.argmax(self.y_probs_sup, axis=1)
