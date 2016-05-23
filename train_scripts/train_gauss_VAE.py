import numpy as np
import cPickle
import os
import sys
import time
from os.path import join as pjoin

import theano
import theano.tensor as T

from models.Gauss_VAE import Gaussian_VAE
from models.neural_net.activation_fns import Sigmoid, Identity, Softmax
from models.neural_net.loss_fns import *
from utils.load_data import load_mnist, load_mnist_w_rotations, load_svhn_pca
from opt_fns import get_adam_updates

### Train & Evaluate ###
def train_and_eval_gaussian_vae(
    dataset,
    hidden_layer_sizes,
    hidden_layer_types,
    latent_size,
    activations,
    prior_mu,
    prior_sigma,
    n_epochs,
    batch_size,
    lookahead,
    adam_lr,
    experiment_dir,
    output_file_base_name,
    random_seed):
    
    rng = np.random.RandomState(random_seed)

    # LOAD DATA
    if "mnist_plus_rot" in dataset:
        datasets = load_mnist_w_rotations(dataset, target_as_one_hot=True, flatten=False, split=(70000, 10000, 20000))
        input_layer_size = 28*28
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        out_activation = Sigmoid
        neg_log_likelihood_fn = calc_binaryVal_negative_log_likelihood
        print "Dataset: MNIST+rot"
    elif "mnist" in dataset:
        # We follow the approach used in [2] to split the MNIST dataset.
        datasets = load_mnist(dataset, target_as_one_hot=True, flatten=True, split=(45000, 5000, 10000))
        input_layer_size = 28*28
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        out_activation = Sigmoid
        neg_log_likelihood_fn = calc_binaryVal_negative_log_likelihood
        print "Dataset: MNIST"
    elif "svhn_pca" in dataset:
        datasets = load_svhn_pca(dataset, target_as_one_hot=True, train_valid_split=(65000, 8257))
        input_layer_size = 500
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        out_activation = Identity
        neg_log_likelihood_fn = calc_realVal_negative_log_likelihood
        print "Dataset: SVHN (PCA reduced)"
    else:
        print "no data found..."
        exit()
    
    train_set_x, _ = datasets[0]
    valid_set_x, _ = datasets[1]
    test_set_x, _ = datasets[2]

    train_set_size = int(train_set_x.shape[0].eval())
    valid_set_size = int(valid_set_x.shape[0].eval())
    test_set_size = int(test_set_x.shape[0].eval())
    print 'Datasets loaded ({:,} train | {:,} valid | {:,} test)'.format(train_set_size, valid_set_size, test_set_size)
    
    # compute number of minibatches for training, validation and testing                                                
    n_train_batches =  train_set_size / batch_size
    n_test_batches = test_set_size / batch_size
    n_valid_batches = valid_set_size / batch_size

    # BUILD MODEL
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x') 

    # construct the Gaussian Variational Autoencoder
    model = Gaussian_VAE(rng=rng, input=x, batch_size=batch_size, layer_sizes=layer_sizes, layer_types=hidden_layer_types, 
                         activations=activations, latent_size=latent_size, out_activation=out_activation)

    # Build the expresson for the cost function.
    data_ll_term = neg_log_likelihood_fn(x, model.x_recon) 
    kl = model.calc_kl_divergence(prior_mu=prior_mu, prior_sigma=prior_sigma)

    # Compose into final costs
    cost = T.mean( data_ll_term + kl )

    updates = get_adam_updates(cost=cost, params=model.params, lr=adam_lr)

    # Compile theano function for testing.
    test_model = theano.function(
        inputs = [index],
        outputs = T.mean(neg_log_likelihood_fn(x, model.x_recon)),
        givens = {x: test_set_x[index * batch_size:(index + 1) * batch_size]})

    # Compile theano function for validation.       
    valid_model = theano.function(
        inputs = [index],
        outputs = T.mean(neg_log_likelihood_fn(x, model.x_recon)),
        givens = {x: valid_set_x[index * batch_size:(index + 1) * batch_size]})

    # Compile theano function for training.
    train_model = theano.function(
        inputs = [index], 
        outputs = [data_ll_term.mean(), kl.mean()],
        updates = updates,
        givens = {x: train_set_x[index * batch_size:(index + 1) * batch_size]})

    # TRAIN MODEL #    
    print 'Training for {} epochs ...'.format(n_epochs)

    best_params = None
    best_valid_error = np.inf
    best_iter = 0
    start_time = time.clock()

    # check if results file already exists, if so, append a number                                                      
    results_file_name = pjoin(experiment_dir, "gauss_vae_results_"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(experiment_dir, "gauss_vae_results_"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)
    results_file = open(results_file_name, 'w')

    stop_training = False
    for epoch_counter in range(n_epochs):
        if stop_training:
            break

        # Train this epoch
        epoch_start_time = time.time()
        avg_training_nll_tracker = 0.
        avg_training_kl_tracker = 0.

        for minibatch_index in xrange(n_train_batches):
            avg_training_nll, avg_training_kl = train_model(minibatch_index)

            # check for NaN, test model anyway even if one is detected                                       
            if (np.isnan(avg_training_nll) or np.isnan(avg_training_kl)):
                print "found NaN...aborting training..."
                results_file.write("found NaN...aborting training... \n\n")
                if epoch_counter > 0:
                    for param, best_param in zip(model.params, best_params):
                        param.set_value(best_param)
                    test_error = sum([test_model(i) for i in xrange(n_test_batches)]) / n_test_batches
                    results = "Ended due to NaN! best epoch {}, best valid error {:.4f}, test error {:.4f}, training time {:.2f}m"
                    results = results.format(best_iter, best_valid_error, test_error, (end_time-start_time)/60)
                    print results
                    results_file.write(results + "\n")
                results_file.close()
                exit()

            avg_training_nll_tracker += avg_training_nll
            avg_training_kl_tracker += avg_training_kl

        epoch_end_time = time.time()

        # Compute some infos about training.
        avg_training_nll_tracker /= (minibatch_index+1)
        avg_training_kl_tracker /= (minibatch_index+1)

        # Compute validation error 
        valid_error = sum([valid_model(i) for i in xrange(n_valid_batches)])/n_valid_batches

        results = "epoch {}, training loss (NLL) {:.4f}, training kl divergence {:.4f}, valid error {:.4f}, time {:.2f} "

        if valid_error < best_valid_error:
            best_iter = epoch_counter
            best_valid_error = valid_error
            results += " ***"
            # Save progression
            best_params = [param.get_value().copy() for param in model.params]
            cPickle.dump(best_params, open(pjoin(experiment_dir, 'gauss_vae_params_'+output_file_base_name+'.pkl'), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        elif epoch_counter-best_iter > lookahead:
            stop_training = True

        # Report and save progress.
        results = results.format(epoch_counter, avg_training_nll_tracker, avg_training_kl_tracker, valid_error, (epoch_end_time-epoch_start_time)/60)
        print results
        results_file.write(results + "\n")
        results_file.flush()

    end_time = time.clock()

    # Reload best model.
    for param, best_param in zip(model.params, best_params):
        param.set_value(best_param)

    # Compute test error on best epoch 
    test_error = sum([test_model(i) for i in xrange(n_test_batches)])/n_test_batches

    results = "Done! best epoch {}, best valid error {:.4f}, test error {:.4f}, training time {:.2f}m"
    results = results.format(best_iter, best_valid_error, test_error, (end_time-start_time)/60)
    print results
    results_file.write(results + "\n")
    results_file.close()

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
