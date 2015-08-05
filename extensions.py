"""
Extensions called during training to generate samples and diagnostic plots and printouts.
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano.tensor as T
import theano

from blocks.extensions import SimpleExtension

import viz
import sampler
import perturb


class PlotSamples(SimpleExtension):
    def __init__(self, model, algorithm, X, path, n_samples=49, 
                 perturbation_kernel=False, **kwargs):
        """
        Generate samples from the model. The do() function is called as an extension during training.
        Generates 4 types of samples:
        - Sample from generative model
        - Sample from image denoising posterior distribution (default signal to noise of 1)
        - Sample from image inpainting posterior distribution (inpaint left half of image)
        - Sample from posterior p^tilde(x0) ~ p(x0) * r(x0)
        """

        super(PlotSamples, self).__init__(**kwargs)
        self.model = model
        self.path = path
        n_samples = np.min([n_samples, X.shape[0]])
        self.X = X[:n_samples].reshape(
            (n_samples, model.n_colors, model.spatial_width, model.spatial_width))
        self.n_samples = n_samples
        self.perturbation_kernel=perturbation_kernel
        X_noisy = T.tensor4('X noisy samp', dtype=theano.config.floatX)
        t = T.matrix('t samp', dtype=theano.config.floatX)
        self.get_mu_sigma = theano.function([X_noisy, t], model.get_mu_sigma(X_noisy, t),
            allow_input_downcast=True)
        perturbation_kernel = False # FIXME, hard coded
        if perturbation_kernel:
            self.logr_grad = perturb.get_logr_grad()
        else:
            # Sets a default value to have r(x) = 0
            self.logr_grad = lambda x: np.zeros_like(self.X).astype(theano.config.floatX)

    def do(self, callback_name, *args):

        import sys
        sys.setrecursionlimit(10000000)

        print "generating samples"
        base_fname_part1 = self.path + '/samples-'
        base_fname_part2 = '_batch%06d'%self.main_loop.status['iterations_done']
        # Basic Sampler
        sampler.generate_samples(self.model, self.get_mu_sigma,
            n_samples=self.n_samples, inpaint=False, denoise_sigma=None, 
            logr_grad = None, X_true=None,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
        # Inpainting
        sampler.generate_samples(self.model, self.get_mu_sigma,
            n_samples=self.n_samples, inpaint=True, denoise_sigma=None, 
            logr_grad = None, X_true=self.X,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
        # Denoising
        sampler.generate_samples(self.model, self.get_mu_sigma,
            n_samples=self.n_samples, inpaint=False, denoise_sigma=1, 
            logr_grad=None, X_true=self.X,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
        # Perturbation Kernel
        sampler.generate_samples(self.model, self.get_mu_sigma, 
            n_samples=self.n_samples, inpaint=False, denoise_sigma=None,
            logr_grad=self.logr_grad, X_true=self.X
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
            



class PlotParameters(SimpleExtension):
    def __init__(self, model, blocks_model, path, **kwargs):
        super(PlotParameters, self).__init__(**kwargs)
        self.path = path
        self.model = model
        self.blocks_model = blocks_model

    def do(self, callback_name, *args):

        import sys
        sys.setrecursionlimit(10000000)

        print "plotting parameters"
        for param in self.blocks_model.parameters:
            param_name = param.name
            filename_safe_name = '-'.join(param_name.split('/')[2:]).replace(' ', '_')
            base_fname_part1 = self.path + '/params-' + filename_safe_name
            base_fname_part2 = '_batch%06d'%self.main_loop.status['iterations_done']
            viz.plot_parameter(param.get_value(), base_fname_part1, base_fname_part2,
                title=param_name, n_colors=self.model.n_colors)


class PlotGradients(SimpleExtension):
    def __init__(self, model, blocks_model, algorithm, X, path, **kwargs):
        super(PlotGradients, self).__init__(**kwargs)
        self.path = path
        self.X = X
        self.model = model
        self.blocks_model = blocks_model
        gradients = []
        for param_name in sorted(self.blocks_model.parameters.keys()):
            gradients.append(algorithm.gradients[self.blocks_model.parameters[param_name]])
        self.grad_f = theano.function(algorithm.inputs, gradients, allow_input_downcast=True)

    def do(self, callback_name, *args):
        print "plotting gradients"
        grad_vals = self.grad_f(self.X)
        keynames = sorted(self.blocks_model.parameters.keys())
        for ii in xrange(len(keynames)):
            param_name = keynames[ii]
            val = grad_vals[ii]
            filename_safe_name = '-'.join(param_name.split('/')[2:]).replace(' ', '_')
            base_fname_part1 = self.path + '/grads-' + filename_safe_name
            base_fname_part2 = '_batch%06d'%self.main_loop.status['iterations_done']
            viz.plot_parameter(val, base_fname_part1, base_fname_part2,
                title="grad " + param_name, n_colors=self.model.n_colors)


class PlotInternalState(SimpleExtension):
    def __init__(self, model, blocks_model, state, features, X, path, **kwargs):
        super(PlotInternalState, self).__init__(**kwargs)
        self.path = path
        self.X = X
        self.model = model
        self.blocks_model = blocks_model
        self.internal_state_f = theano.function([features], state, allow_input_downcast=True)
        self.internal_state_names = []
        for var in state:
            self.internal_state_names.append(var.name)

    def do(self, callback_name, *args):
        print "plotting internal state of network"
        state = self.internal_state_f(self.X)
        for ii in xrange(len(state)):
            param_name = self.internal_state_names[ii]
            val = state[ii]
            filename_safe_name = param_name.replace(' ', '_').replace('/', '-')
            base_fname_part1 = self.path + '/state-' + filename_safe_name
            base_fname_part2 = '_batch%06d'%self.main_loop.status['iterations_done']
            viz.plot_parameter(val, base_fname_part1, base_fname_part2,
                title="state " + param_name, n_colors=self.model.n_colors)


class PlotMonitors(SimpleExtension):
    def __init__(self, path, burn_in_iters=0, **kwargs):
        super(PlotMonitors, self).__init__(**kwargs)
        self.path = path
        self.burn_in_iters = burn_in_iters

    def do(self, callback_name, *args):
        print "plotting monitors"
        try:
            df = self.main_loop.log.to_dataframe()
        except AttributeError:
            # This starting breaking after a Blocks update.
            print "Failed to generate monitoring plots due to Blocks interface change."
            return
        iter_number  = df.tail(1).index
        # Throw out the first burn_in values
        # as the objective is often much larger
        # in that period.
        if iter_number > self.burn_in_iters:
            df = df.loc[self.burn_in_iters:]
        cols = [col for col in df.columns if col.startswith(('cost', 'train', 'test'))]
        df = df[cols].interpolate(method='linear')

        # If we don't have any non-nan dataframes, don't plot
        if len(df) == 0:
            return
        try:
            axs = df.interpolate(method='linear').plot(
                subplots=True, legend=False, figsize=(5, len(cols)*2))
        except TypeError:
            # This starting breaking after a different Blocks update.
            print "Failed to generate monitoring plots due to Blocks interface change."
            return

        for ax, cname in zip(axs, cols):
            ax.set_title(cname)
        fn = os.path.join(self.path,
            'monitors_subplots_batch%06d.png' % self.main_loop.status['iterations_done'])
        plt.savefig(fn, bbox_inches='tight')

        plt.clf()
        df.plot(subplots=False, figsize=(15,10))
        plt.gcf().tight_layout()
        fn = os.path.join(self.path,
            'monitors_batch%06d.png' % self.main_loop.status['iterations_done'])
        plt.savefig(fn, bbox_inches='tight')
        plt.close('all')


def decay_learning_rate(iteration, old_value):
    # TODO the numbers in this function should not be hard coded

    # this is called every epoch
    # reduce the learning rate by 10 every 1000 epochs
    min_value = 1e-4

    decay_rate = np.exp(np.log(0.1)/1000.)
    new_value = decay_rate*old_value
    if new_value < min_value:
        new_value = min_value
    print "learning rate %g"%new_value
    return np.float32(new_value)
