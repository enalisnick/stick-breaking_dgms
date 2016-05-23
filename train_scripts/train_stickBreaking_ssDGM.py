import numpy as np
import cPickle
import os
import sys
import time
from os.path import join as pjoin

import theano
import theano.tensor as T

from models.ss_StickBreaking_DGM import SS_StickBreaking_DGM
from models.neural_net.activation_fns import Sigmoid, Identity, Softmax
from models.neural_net.loss_fns import *
from utils.load_data import load_mnist, load_mnist_w_rotations, load_svhn_pca
from opt_fns import get_adam_updates

### Train & Evaluate ###
def train_and_eval_stickBreaking_ss_dgm(
    dataset,
    drop_p,
    hidden_layer_sizes,
    hidden_layer_types,
    latent_size,
    activations,
    supervised_weight,
    prior_alpha,
    prior_beta,
    n_epochs,
    unsup_batch_size,
    lookahead,
    adam_lr,
    experiment_dir,
    output_file_base_name,
    random_seed):
    
    rng = np.random.RandomState(random_seed)

    # LOAD DATA
    if "mnist_plus_rot" in dataset:
        datasets = load_mnist_w_rotations(dataset, target_as_one_hot=True, flatten=False, split=(70000, 10000, 20000), drop_percentage=drop_p)
        input_layer_size = 28*28
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        output_size = 10
        out_activation = Sigmoid
        label_fn = Softmax
        neg_log_likelihood_fn = calc_binaryVal_negative_log_likelihood
        print "Dataset: MNIST+rot"
    elif "mnist" in dataset:
        # We follow the approach used in [2] to split the MNIST dataset.
        datasets = load_mnist(dataset, target_as_one_hot=True, flatten=True, split=(45000, 5000, 10000), drop_percentage=drop_p)
        input_layer_size = 28*28
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        output_size = 10
        out_activation = Sigmoid
        label_fn = Softmax
        neg_log_likelihood_fn = calc_binaryVal_negative_log_likelihood
        print "Dataset: MNIST"
    elif "svhn" in dataset:
        datasets = load_svhn_pca(dataset, target_as_one_hot=True, train_valid_split=(65000, 8257), drop_percentage=drop_p)
        input_layer_size = 500
        layer_sizes = [input_layer_size] + hidden_layer_sizes
        output_size = 10
        out_activation = Identity
        label_fn = Softmax
        neg_log_likelihood_fn = calc_realVal_negative_log_likelihood
        print "Dataset: SVHN (PCA reduced)"
    else:
        print "no data found..."
        exit()
    
    train_set_x_sup, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    train_set_x_unsup, _ = datasets[2]
    test_set_x, test_set_y = datasets[3]

    sup_train_set_size = int(train_set_x_sup.shape[0].eval())
    unsup_train_set_size = int(train_set_x_unsup.shape[0].eval())
    valid_set_size = int(valid_set_x.shape[0].eval())
    test_set_size = int(test_set_x.shape[0].eval())
    percent_unsupervised = unsup_train_set_size / float(sup_train_set_size + unsup_train_set_size) 
    print 'Datasets loaded ({:,} sup. train | {:,} unsup. train | {:,} valid | {:,} test)'.format(sup_train_set_size, unsup_train_set_size, valid_set_size, test_set_size)
    
    # compute number of minibatches for training, validation and testing                                                
    n_train_batches =  unsup_train_set_size / unsup_batch_size
    sup_batch_size = sup_train_set_size / n_train_batches
    if sup_batch_size < 1:
        print "not enough expamples w/ labels...increase (unsupervised) batch size..."
        exit()
    n_test_batches = test_set_size / sup_batch_size
    n_valid_batches = valid_set_size / sup_batch_size

    # BUILD MODEL
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x_sup = T.matrix('x_sup')  
    x_un_sup = T.matrix('x_un_sup')
    y = T.matrix('y')

    # construct the Gaussian semi-supervised DGM
    model = SS_StickBreaking_DGM(rng=rng, sup_input=x_sup, un_sup_input=x_un_sup, labels=y, 
                            sup_batch_size=sup_batch_size, un_sup_batch_size=unsup_batch_size, 
                            layer_sizes=layer_sizes, layer_types=hidden_layer_types, activations=activations, 
                            label_size=output_size, latent_size=latent_size, out_activation=out_activation, label_fn=label_fn)

    # Build the expresson for the cost function.
    # SUPERVISED TERMS
    supervised_data_ll_term = neg_log_likelihood_fn(x_sup, model.x_recon_sup)
    supervised_label_ll_term = calc_categoricalVal_negative_log_likelihood(y, model.y_probs_sup)
    supervised_kl = model.calc_sup_kl_divergence(prior_alpha=prior_alpha, prior_beta=prior_beta)
    # UNSUPERVISED TERMS
    un_supervised_data_ll_term = T.sum(model.y_probs_un_sup * neg_log_likelihood_fn(x_un_sup, model.x_recon_un_sup, axis_to_sum=2), axis=1)
    label_entropy_term = calc_cat_entropy(model.y_probs_un_sup)
    un_supervised_kl = model.calc_un_sup_kl_divergence(prior_alpha=prior_alpha, prior_beta=prior_beta)

    # Compose into final costs
    supervised_cost = T.mean(supervised_data_ll_term + supervised_kl + supervised_label_ll_term)
    un_supervised_cost = T.mean(un_supervised_data_ll_term + un_supervised_kl - label_entropy_term)
    cost = supervised_weight * supervised_cost + (1-supervised_weight) * un_supervised_cost

    updates = get_adam_updates(cost=cost, params=model.params, lr=adam_lr)

    # Compile theano function for testing.
    test_model = theano.function(
        inputs = [index],
        outputs = calc_prediction_errors(T.argmax(y,axis=1), model.y_preds_sup),
        givens = {x_sup: test_set_x[index * sup_batch_size:(index + 1) * sup_batch_size],
                  y: test_set_y[index * sup_batch_size:(index + 1) * sup_batch_size]})

    # Compile theano function for validation.       
    valid_model = theano.function(
        inputs = [index],
        outputs = calc_prediction_errors(T.argmax(y,axis=1), model.y_preds_sup),
        givens = {x_sup: valid_set_x[index * sup_batch_size:(index + 1) * sup_batch_size],
                  y: valid_set_y[index * sup_batch_size:(index + 1) * sup_batch_size]})

    # Compile theano function for training.
    train_model = theano.function(
        inputs = [index], 
        outputs = [supervised_data_ll_term.mean(), un_supervised_data_ll_term.mean(), supervised_kl.mean(), un_supervised_kl.mean()],
        updates = updates,
        givens = {x_sup: train_set_x_sup[index * sup_batch_size:(index + 1) * sup_batch_size],
                  x_un_sup: train_set_x_unsup[index * unsup_batch_size:(index + 1) * unsup_batch_size],
                  y: train_set_y[index * sup_batch_size:(index + 1) * sup_batch_size]})

    # TRAIN MODEL #    
    print 'Training for {} epochs ...'.format(n_epochs)

    best_params = None
    best_valid_error = np.inf
    best_iter = 0
    start_time = time.clock()

    # check if results file already exists, if so, append a number
    results_file_name = pjoin(experiment_dir, "sb_ss-dgm_results_"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(experiment_dir, "sb_ss-dgm_results_"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)
    results_file = open(results_file_name, 'w')

    stop_training = False
    for epoch_counter in range(n_epochs):
        if stop_training:
            break

        # Train this epoch
        epoch_start_time = time.time()
        avg_training_sup_nll_tracker = 0.
        avg_training_sup_kl_tracker = 0.
        avg_training_unsup_nll_tracker = 0.
        avg_training_unsup_kl_tracker = 0.

        for minibatch_index in xrange(n_train_batches):
            avg_training_sup_nll, avg_training_unsup_nll, avg_training_sup_kl, avg_training_unsup_kl = train_model(minibatch_index)

            # check for NaN, test model anyway even if one is detected 
            if (np.isnan(avg_training_sup_nll) or np.isnan(avg_training_unsup_nll) or np.isnan(avg_training_sup_kl) or np.isnan(avg_training_unsup_kl)):
                print "found NaN...aborting training..."
                results_file.write("found NaN...aborting training... \n\n")
                if epoch_counter > 0:
                    for param, best_param in zip(model.params, best_params):
                        param.set_value(best_param)
                    test_nb_errors = sum([test_model(i) for i in xrange(n_test_batches)])
                    test_error = test_nb_errors / float(test_set_size)
                    results = "Ended due to NaN! best epoch {}, best valid error {:.2%} ({:,}), test error {:.2%} ({:,})"
                    results = results.format(best_iter, best_valid_error, best_valid_error*valid_set_size, test_error, test_nb_errors)
                    print results
                    results_file.write(results + "\n")
                results_file.close()
                exit()

            avg_training_sup_nll_tracker += avg_training_sup_nll
            avg_training_unsup_nll_tracker += avg_training_unsup_nll
            avg_training_sup_kl_tracker += avg_training_sup_kl
            avg_training_unsup_kl_tracker += avg_training_unsup_kl

        epoch_end_time = time.time()

        # Compute some infos about training.
        avg_training_sup_nll_tracker /= (minibatch_index+1)
        avg_training_unsup_nll_tracker /= (minibatch_index+1)
        avg_training_sup_kl_tracker /= (minibatch_index+1)
        avg_training_unsup_kl_tracker /= (minibatch_index+1)

        # Compute validation error 
        valid_nb_errors = sum([valid_model(i) for i in xrange(n_valid_batches)])
        valid_error = valid_nb_errors / float(valid_set_size)

        results = "epoch {}, (supervised training loss (NLL) {:.4f}, unsupervised training loss (NLL) {:.4f}), (supervised training kl-div {:.4f}, unsupervised training kl-div {:.4f}), valid error {:.2%} ({:,}), time {:.2f} "

        if valid_error < best_valid_error:
            best_iter = epoch_counter
            best_valid_error = valid_error
            results += " ***"
            # Save progression
            best_params = [param.get_value().copy() for param in model.params]
            # shouldn't need to check if file exists
            cPickle.dump(best_params, open(pjoin(experiment_dir, 'sb_ss-dgm_params_'+output_file_base_name+'.pkl'), 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        elif epoch_counter-best_iter > lookahead:
            stop_training = True

        # Report and save progress.
        results = results.format(epoch_counter, avg_training_sup_nll_tracker, avg_training_unsup_nll_tracker, avg_training_sup_kl_tracker, avg_training_unsup_kl_tracker, valid_error, valid_nb_errors, (epoch_end_time-epoch_start_time)/60)
        print results
        results_file.write(results + "\n")
        results_file.flush()

    end_time = time.clock()

    # Reload best model.
    for param, best_param in zip(model.params, best_params):
        param.set_value(best_param)

    # Compute test error on best epoch 
    test_nb_errors = sum([test_model(i) for i in xrange(n_test_batches)])
    test_error = test_nb_errors / float(test_set_size)

    results = "Done! best epoch {}, best valid error {:.2%} ({:,}), test error {:.2%} ({:,}), training time {:.2f}m"
    results = results.format(best_iter, best_valid_error, best_valid_error*valid_set_size, test_error, test_nb_errors, (end_time-start_time)/60)
    print results
    results_file.write(results + "\n")
    results_file.close()

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
