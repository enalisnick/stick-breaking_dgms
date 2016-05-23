### Deep Generative Models with Stick-Breaking Priors
This reposiory contains [Theano](https://github.com/Theano) implementations of the models described in [*Deep Generative Models with Stick-Breaking Priors*](http://arxiv.org/abs/1605.06197).  Documentation and development still in progress.   

#### Stick-Breaking Variational Autoencoder
The [Stick-Breaking Autoencoder](https://github.com/enalisnick/stick-breaking_dgms/blob/master/models/StickBreaking_VAE.py) is a nonparametric reformulation of the [Variational Autoencoder](https://arxiv.org/abs/1312.6114).  The latent variables are drawn from a [stick-breaking process](http://blog.shakirm.com/2015/12/machine-learning-trick-of-the-day-6-tricks-with-sticks/), a combinatorial mechanism for sampling from an infinite distribution.  This implementation uses a truncated variational approximation; see the paper for discussion on un-truncated approaches.  The model's feedforward architecture can be seen below.  Note the cross-dependencies that are a by-product of the stick-breaking process' recursive nature.  
<img src="http://www.ics.uci.edu/~enalisni/SBAE_arch.png" width="250">

The Stick-Breaking Autoencoder can be trained on MNIST by running:

    python run_stickBreaking_VAE_experiments.py
Be sure to set the Theano flags appropriately for GPU usage.  Running with the option ```--help``` shows command line arguments for changing the dataset, architecure, and other hyperparameters.

#### Semi-Supervised Variant
Fully- and semi-supervised classification experiments can be run with the [semi-supervised deep generative model with stick-breaking latent variables](https://github.com/enalisnick/stick-breaking_dgms/blob/master/models/ss_StickBreaking_DGM.py), a nonparametric reformulation of [Kingma et al's M2 model](http://arxiv.org/abs/1406.5298).  This model is similar to the variational autoencoder, the change being that a class label is introduced as another latent variable that is marginalized when unobserved.  The model's feedforward architecture is diagrammed below.  
<img src="http://www.ics.uci.edu/~enalisni/NP-SSDGM_arch.png" width="250">

The semi-supervised deep generative model can be trained on MNIST by running:

    python run_stickBreaking_ssDGM_experiments.py
Again, be sure to set the Theano flags appropriately for GPU usage, and use the ```--help``` option to see command line arguments.

#### Computational Issues to Note
There are a few issues to be aware of when training these stick-breaking models.  The first is that one term in the KL divergence between the prior's Beta distributions and the posterior's Kumaraswamy variables needs a Taylor series approximation.  [See the supplemental materials for the derivation](http://www.ics.uci.edu/~enalisni/sb_dgm_supp_mat.pdf).  The code computes this approximation with the leading ten terms (hard coded).

The second issue is that calculating the KL divergence involves the [digamma function](https://en.wikipedia.org/wiki/Digamma_function).  The problem is that its derivative, the [polygamma function](https://en.wikipedia.org/wiki/Polygamma_function), is not easy to implement in C, which is necessary for using the GPU.  [Here](https://sourceforge.net/p/mcmc-jags/code-0/ci/default/tree/src/jrmath/polygamma.c) is the only C code I'm aware of that is capable of computing the polygamma, and I haven't had the time to add the functionality to the Theano code base.  As a result, this code employs another Taylor series approximation, expanded around zero, to compute the digamma function.  For large values of *b* (Kumaraswamy parameter), the approximation gets bad, but this should not be an issue if the prior's concentration parameter (the Beta's beta value) is set to a reasonably small value.  Due to these two approximations, it's possible (but rare) to have a negative KL divergence. 

Lastly, if the posterior's truncation level is set too low, sampling the Kumaraswamy variables can cause NaNs.  This occurs because the model tries to perform a hard thresholding of the latent representation, and to do this, the variational Kumaraswamy parameters must be set to very small or very large values.  Very large values cause the Taylor approximations to become inaccurate, and very small values can cause the 1/a, 1/b terms to go to infinity.  If NaNs are encountered, increasing the truncation level or clipping the parameters of the variational Kumaraswamys usually solves the problem. 

#### Roadmap 
I plan to add models with convolutional layers and with multiple stochastic layers.
