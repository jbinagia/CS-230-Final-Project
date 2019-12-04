import numpy as np
import torch

from .system import System

#################################################################################

class DoubleWell(System):
    """
    Simple double well potential as used in Noe et al. (2019).
    """

    def __init__(self, params = None, **kwargs):
        params_default = {
                "a" : 1.0,
                "b" : 6.0,
                "c" : 1.0,
                "d" : 1.0
            }

        # Set Parameters
        if (params != None):
            self.params = params
        else:
            self.params = params_default

    def energy(self, x):
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        d = self.params['d']
        if (len(x.shape) > 1):
           return a*x[:,0]**4/4 - b*x[:,0]**2/2 + c*x[:,0] + d*x[:,1]**2/2
        else:
            return a*x[0]**4/4 - b*x[0]**2/2 + c*x[0] + d*x[1]**2/2
