import numpy as np
import tensorflow as tf

from system import System

#################################################################################

class IsingModel(System):

    params_default = {'a4' : 1.0,
                      'a2' : 6.0,
                      'a1' : 1.0,
                      'k' : 1.0,
                      'dim' : 1}

    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.dim = self.params['dim']

    def energy(self, x):
        pass

    def energy_tf(self, x):
        pass