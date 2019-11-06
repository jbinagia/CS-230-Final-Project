import numpy as np
import torch

from .system import System

#################################################################################

class SHOModel(System):
    """
    Toy simple harmonic oscillator model, two particles connected by a spring
    stuck in a 1D box of length L.
    """

    params_default = {
        "k" : 1.0,
        "L" : 4.0,
    }

    def __init__(self, params = None, **kwargs):
        # Set Parameters
        super().__init__(params, **kwargs)
        if (params != None):
            self.params = params
        else:
            self.params = params_default
        # Init parent class
        super().__init__(params, **kwargs)

    def init_coords(self, N, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        shape = [N] * 2
        return 2 * np.random.randint(0, 2, size = shape) - 1

    def energy(self, x):
        return self.params['k']*np.power(x[:,1]-x[:,0],2) 
