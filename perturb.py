"""
We can perturb the diffusion kernel by an arbitrary function r(x), so that
we now sample from p^tilde(x) = (1 / Z) r(x) p(x). Here, we prepare such a function
where r(x) is a classifier of the form r(x) = p(y_0|x)
"""

