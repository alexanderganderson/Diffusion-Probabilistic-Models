"""
The goal of this script is to load the .pkl version of a DPM model and run a
sampler

Requires the following:

(1) There exists a file models/dataset_classifier.zip that contains the
result from training a classifier from perturb.py

(2) There exists a file models/model_dataset.pkl that contains the main
    mainloop from running train.py. Note that if different extensions were
    run, then the line extracting out the plot samples extension might fail.
"""

import numpy as np
from argparse import ArgumentParser

from theano.misc import pkl_utils
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift

import sampler


parser = ArgumentParser("Perturb a DPM")

parser.add_argument('--dataset', type=str, default='IMAGENET',
                    help='Name of dataset to use.')
parser.add_argument('--save_path', type=str, default='output',
                    help='Path to save output')
parser.add_argument('--n_samples', type=int, default=4,
                    help='Number of samples to generate')

args = parser.parse_args()
dataset, save_path = args.dataset, args.save_path

dpm_fn = 'models/model_{}.pkl'.format(dataset)
n_samples = 4
batch_size = 200  # For computing normalization for dataset
labels = [105]

# Load the DPM (Saved as a Blocks MainLoop)
with open(dpm_fn, 'r') as f:
    main_loop = pkl_utils.load(f)

print "generating samples"
base_fname_part1 = save_path + '/samples-' + dataset + '-'
base_fname_part2 = '_batch%06d' % main_loop.status['iterations_done']

# FIXME: eww, we want PlotSamples extensions, better way to get it
plotsamples_ext = main_loop.extensions[6]

model = plotsamples_ext.model


def get_dataset(dataset):
    """
    Returns
    -------
    dataset_train : fuel dataset
        Training set
    dataset_test : fuel dataset
        Test set
    """
    if dataset == 'MNIST':
        from fuel.datasets import MNIST
        return MNIST(['train']), MNIST(['test'])
    elif dataset == 'CIFAR10':
        from fuel.datasets import CIFAR10
        return CIFAR10(['train']), CIFAR10(['test'])
    elif dataset == 'IMAGENET':
        from imagenet_data import IMAGENET
        return (IMAGENET(['train'], width=model.spatial_width),
                IMAGENET(['test'], width=model.spatial_width))
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

(dataset_train, dataset_test) = get_dataset(dataset)

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

# Generate Samples with a perturbation
for i in labels:
    if dataset == 'IMAGENET':
        import imagenet_perturb
        r, logr_grad = imagenet_perturb.get_logr_grad(
            dataset, shft, scl, model.spatial_width, label=i)
    else:
        import perturb
        from perturb import ConvMLP  # Strange thing with blocks
        r, logr_grad = perturb.get_logr_grad(dataset, label=i)

    X0 = sampler.generate_samples(
        model, get_mu_sigma,
        n_samples=n_samples, inpaint=False,
        denoise_sigma=None,
        logr_grad=logr_grad, X_true=X,
        base_fname_part1=base_fname_part1+'label%02d' % i,
        base_fname_part2=base_fname_part2)
    print r(X0)

# Generate baseline samples
sampler.generate_samples(model, get_mu_sigma,
                         n_samples=n_samples, inpaint=False,
                         denoise_sigma=None,
                         logr_grad=None, X_true=None,
                         base_fname_part1=base_fname_part1,
                         base_fname_part2=base_fname_part2)
