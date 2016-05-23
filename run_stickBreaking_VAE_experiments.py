import argparse
import os
from os.path import join as pjoin
import theano

from models.neural_net.layers import HiddenLayer, ResidualHiddenLayer
from models.neural_net.activation_fns import ReLU, Sigmoid, Identity
from train_scripts.train_gauss_ssDGM import train_and_eval_stickBreaking_vae
from utils.utils import mkdirs

def build_argparser():
    DESCRIPTION = ("Train a Stick-Breaking Variational Autoencoder using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    dataset = p.add_argument_group("Experiment options")
    dataset.add_argument('--dataset', default="mnist", choices=["mnist", "mnist_plus_rot", "svhn_pca"],
                         help="either 'mnist' or 'mnist_plus_rot' or 'svhn_pca'. Default:%(default)s")

    model = p.add_argument_group("Model options")
    model.add_argument('--hidden-type', default="traditional", choices=["traditional", "residual"],
                         help='either traditional or residual. Default:%(default)s')
    model.add_argument('--nb-hidden-layers', type=int, default=1,
                         help='number of hidden layers in the encoder/decoder. Default:%(default)s')
    model.add_argument('--skip', type=int, default=0,
                         help='number of hidden layers that have skip connections. Default:%(default)s')
    model.add_argument('--hidden-size', type=int, default=500,
                         help='number of units in each hidden layer. Default:%(default)s')
    model.add_argument('--activation-type', default="relu", choices=["relu", "sigmoid"],
                         help='either relu or sigmoid. Default:%(default)s')
    model.add_argument('--truncation-level', type=int, default=50,
                         help='Maximum dimensionality of latent variable. Default:%(default)s')
     model.add_argument('--alpha0', type=float, default=5.,
                         help="the Beta prior's concentration parameter: v ~ Beta(1, alpha0). The larger the alpha0, the more latent variables. Default:%(default)s")

    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int, default=100,
                          help='size of the batch to use when training the model. Default: %(default)s.')
    training.add_argument('--max-epoch', type=int, metavar='N', default=2000,
                          help='train for a maximum of N epochs. Default: %(default)s')
    training.add_argument('--lookahead', type=int, metavar='K', default=100,
                          help='use early stopping with a lookahead of K. Default: %(default)s')
    # training.add_argument('--clip-gradient', type=float,
    #                       help='if provided, gradient norms will be clipped to this value (if it exceeds it).')

    optimizer = p.add_argument_group("AdaM Options")
    optimizer.add_argument('--learning-rate', type=float, default=.0003,
                         help="the AdaM learning rate (alpha) parameter. Default:%(default)s")     
    #optimizer = optimizer.add_mutually_exclusive_group(required=True)
    #optimizer.add_argument('--Adam', metavar="[LR=0.0003]", type=str, help='use Adam for training.')

    general = p.add_argument_group("General arguments")
    general.add_argument('--experiment-dir', default="./experiments/",
                         help='name of the folder where to save the experiment. Default: %(default)s.')

    return p


if __name__ == '__main__':
    random_seed = 1234

    # get command-line args
    parser = build_argparser()
    args = parser.parse_args()
    args_dict = vars(args)
    args_string = ''.join('{}_{}_'.format(key, val) for key, val in sorted(args_dict.items()) if key not in ['experiment_dir','lookahead','max_epoch'])[:-1]

    # Check the parameters are correct.
    if args.nb_hidden_layers < args.skip and args.nb_hidden_layers > 0:
        raise ValueError("Nb. hiddens layers should be (at least) one less than number of skip connections.")

    print "Using Theano v.{}".format(theano.version.short_version)

    # ARCHITECTURE PARAMS
    hidden_layer_sizes = [args.hidden_size] * args.nb_hidden_layers

    if args.activation_type == "relu":
        activations = [ReLU] * args.nb_hidden_layers
    elif args.activation_type == "sigmoid":
        activations = [Sigmoid] * args.nb_hidden_layers

    if args.hidden_type == "traditional":
        hidden_layer_types = [HiddenLayer] * args.nb_hidden_layers
    elif args.hidden_type == "residual":
        hidden_layer_types = [HiddenLayer] * (args.nb_hidden_layers - args.skip) + [ResidualHiddenLayer] * (args.skip)

    # P(\pi) PARAMS          
    prior_alpha = 1.
    prior_beta = args.alpha0

    # DATA PARAMS
    # Create datasets and experiments folders is needed.
    dataset_dir = mkdirs("./datasets")
    mkdirs(args.experiment_dir)

    dataset_name = None
    
    if args.dataset == 'mnist_plus_rot' or args.dataset == 'svhn_pca':
        dataset = pjoin(dataset_dir, args.dataset + ".pkl")
    else:
        dataset = pjoin(dataset_dir, args.dataset + ".npz")


    print "Datasets dir: {}".format(os.path.abspath(dataset_dir))
    print "Experiment dir: {}".format(os.path.abspath(args.experiment_dir))

    train_and_eval_stickBreaking_vae(
        dataset=dataset,
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_layer_types = hidden_layer_types,
        latent_size=args.truncation_level,
        activations=activations,
        prior_alpha = prior_alpha,
        prior_beta = prior_beta,
        n_epochs=args.max_epoch,
        batch_size=args.batch_size,
        lookahead=args.lookahead,
        adam_lr=args.learning_rate,
        experiment_dir=args.experiment_dir,
        output_file_base_name = args_string,
        random_seed=random_seed)
