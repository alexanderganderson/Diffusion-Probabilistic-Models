"""
The goal of this script is to load the .pkl version of a model and run a sampler
"""
import cPickle as pkl
import theano.tensor as T
import theano

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift

import perturb
import sampler
import util

fn = 'models/model_cifar.pkl'
save_path = 'output'
n_samples = 49
batch_size = 200
with open(fn, 'r') as f:
    main_loop = pkl.load(f)

print "generating samples"
base_fname_part1 = save_path + '/samples-'
base_fname_part2 = '_batch%06d'%main_loop.status['iterations_done']

# FIXME: eww, we want PlotSamples extensions
plotsamples_ext = main_loop.extensions[6]

model = plotsamples_ext.model

from fuel.datasets import CIFAR10
dataset_train = CIFAR10(['train'], sources=('features',))
dataset_test = CIFAR10(['test'], sources=('features',))
n_colors = 3
spatial_width = 32

train_stream = Flatten(DataStream.default_stream(dataset_train,
                              iteration_scheme=ShuffledScheme(
                                  examples=dataset_train.num_examples,
                                  batch_size=batch_size)))
test_stream = Flatten(DataStream.default_stream(dataset_test,
                             iteration_scheme=ShuffledScheme(
                                 examples=dataset_test.num_examples,
                                 batch_size=batch_size)))

# make the training data 0 mean and variance 1
Xbatch = next(train_stream.get_epoch_iterator())[0]
scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
shft = -np.mean(Xbatch*scl)
# scale is applied before shift
test_stream = ScaleAndShift(test_stream, scl, shft)

X = next(test_stream.get_epoch_iterator())[0]
n_samples = np.min([n_samples, X.shape[0]])


X = X[:n_samples].reshape(
    (n_samples, model.n_colors, model.spatial_width, model.spatial_width))



#X_noisy = T.tensor4('X noisy samp', dtype=theano.config.floatX)
#t = T.matrix('t samp', dtype=theano.config.floatX)
#get_mu_sigma = theano.function([X_noisy, t], model.get_mu_sigma(X_noisy, t),
#                               allow_input_downcast=True)

get_mu_sigma = plotsamples_ext.get_mu_sigma


r, logr_grad = perturb.get_logr_grad()

# Sets a default value to have r(x) = 0
#    self.r = lambda x: np.zeros((self.X.shape[0],))
#    self.logr_grad = lambda x: np.zeros_like(self.X).astype(theano.config.floatX)


sampler.generate_samples(model, get_mu_sigma, 
               n_samples=n_samples, inpaint=False, denoise_sigma=None,
               logr_grad=self.logr_grad, X_true=self.X,
               base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)

sampler.generate_samples(model, get_mu_sigma,
            n_samples=n_samples, inpaint=False, denoise_sigma=None, 
            logr_grad = None, X_true=None,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
