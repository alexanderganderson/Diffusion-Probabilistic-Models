"""
The goal of this script is to load the .pkl version of a model and run a
sampler

Requires the following:

(1) There exists a file models/dataset_classifier.zip that contains the
result from training a classifier from perturb.py

(2) There exists a file models/model_dataset.pkl that contains the main
    mainloop from running train.py. Note that if different extensions were
    run, then the line extracting out the plot samples extension might fail.
"""
# import theano.tensor as T
from theano.misc import pkl_utils
import theano
import numpy as np
from argparse import ArgumentParser

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift

import perturb
from perturb import ConvMLP
import sampler
import util

parser = ArgumentParser("An example of training a convolutional network ")

parser.add_argument('--dataset', type=str, default='MNIST',
                    help='Name of dataset to use.')
args = parser.parse_args()
dataset = args.dataset

mainloop_fn = 'models/model_' + dataset + '.pkl'
save_path = 'output'
n_samples = 49
batch_size = 200

with open(mainloop_fn, 'r') as f:
    main_loop = pkl_utils.load(f)

print "generating samples"
base_fname_part1 = save_path + '/samples-' + dataset + '-'
base_fname_part2 = '_batch%06d' % main_loop.status['iterations_done']

# FIXME: eww, we want PlotSamples extensions, better way to get it
plotsamples_ext = main_loop.extensions[6]

model = plotsamples_ext.model

if dataset == 'MNIST':
    from fuel.datasets import MNIST
    dataset_train = MNIST(['train'])
    dataset_test = MNIST(['test'])
    n_colors = 1
    spatial_width = 28
elif dataset == 'CIFAR10':
    from fuel.datasets import CIFAR10
    dataset_train = CIFAR10(['train'])
    dataset_test = CIFAR10(['test'])
    n_colors = 3
    spatial_width = 32
elif dataset == 'IMAGENET':
    from imagenet_data import IMAGENET
    spatial_width = 128
    dataset_train = IMAGENET(['train'], width=spatial_width)
    dataset_test = IMAGENET(['test'], width=spatial_width)
    n_colors = 3
else:
    raise ValueError("Unknown dataset %s." % args.dataset)

train_stream = Flatten(DataStream.default_stream(
    dataset_train,
    iteration_scheme=ShuffledScheme(
        examples=dataset_train.num_examples,
        batch_size=batch_size)))
test_stream = Flatten(DataStream.default_stream(
    dataset_test,
    iteration_scheme=ShuffledScheme(
        examples=dataset_test.num_examples,
        batch_size=batch_size)))

# make the training data 0 mean and variance 1
Xbatch = next(train_stream.get_epoch_iterator())[0]
scl = 1. / np.sqrt(np.mean((Xbatch - np.mean(Xbatch))**2))
shft = -np.mean(Xbatch * scl)
# scale is applied before shift
test_stream = ScaleAndShift(test_stream, scl, shft)

X = next(test_stream.get_epoch_iterator())[0]
n_samples = np.min([n_samples, X.shape[0]])


X = X[:n_samples].reshape(
    (n_samples, model.n_colors, model.spatial_width, model.spatial_width))


# X_noisy = T.tensor4('X noisy samp', dtype=theano.config.floatX)
# t = T.matrix('t samp', dtype=theano.config.floatX)
# get_mu_sigma = theano.function([X_noisy, t], model.get_mu_sigma(X_noisy, t),
#                               allow_input_downcast=True)

get_mu_sigma = plotsamples_ext.get_mu_sigma


# Generate the samples with nothing special
# sampler.generate_samples(model, get_mu_sigma,
#                          n_samples=n_samples, inpaint=False,
#                          denoise_sigma=None,
#                          logr_grad=None, X_true=None,
#                          base_fname_part1=base_fname_part1,
#                          base_fname_part2=base_fname_part2)


# Generate Samples with a perturbation
for i in [6]:  # range(10):
    r, logr_grad = perturb.get_logr_grad(dataset, label=i)
    X0 = sampler.generate_samples(
        model, get_mu_sigma,
        n_samples=n_samples, inpaint=False,
        denoise_sigma=None,
        logr_grad=logr_grad, X_true=X,
        base_fname_part1=base_fname_part1+'label%02d' % i,
        base_fname_part2=base_fname_part2)
    print r(X0)
