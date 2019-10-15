import itertools
import numpy as np
import tensorflow as tf

from .system import System
from .pbc import Box
from .potentials import lj_potential, harmonic_potential

#################################################################################

class IsingModel(System):
    """
    A 2-dimensional Ising model grid.
    """

    params_default = {
        "h" : 0.0,
        "J" : 1.0
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)

    def init_coords(self, N, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        shape = [N] * 2
        return 2 * np.random.randint(0, 2, size = shape) - 1

    def calc_energy(self, x):
        N = x.shape[0]

        en = 0.0    
        for (i, j) in itertools.product(range(N), range(N)):
            sij = x[i,j]
            nb = x[(i+1)%N, j] + x[i, (j+1)%N] + x[(i-1)%N, j] + x[i, (j-1)%N]
            en += -0.25*self.params["J"]*nb*sij
            en += self.params["h"]*sij # Linear coupling to each lattice index

        return en

    def calc_energy_idx(self, x, idx):
        N = x.shape[0]
        i, j = np.unravel_index(idx, (N, N))

        sij = x[i,j]
        nb = x[(i+1)%N, j] + x[i, (j+1)%N] + x[(i-1)%N, j] + x[i, (j-1)%N]
        eidx = -0.5*self.params["J"]*nb*sij 
        eidx += self.params["h"]*sij

        return eidx

    def displace(self, x, idx):
        N = x.shape[0]
        i, j = np.unravel_index(idx, (N, N))
        return -1*x[i,j]

    def oprm(self, x):
        """Order parameter for IsingModel is the average magnetization."""
        return np.mean(x)

    #################################################################################

    def draw_config(self, x):
        import matplotlib.pyplot as plt

        (fig, ax) = plt.subplots(1, figsize = (5, 5))

        N = x.shape[0]
        X, Y = np.meshgrid(range(N+1), range(N+1)) 

        plt.setp(ax.get_yticklabels(), visible = False)
        plt.setp(ax.get_xticklabels(), visible = False)     
        plt.pcolormesh(X, Y, -1*x, cmap = plt.cm.RdBu)