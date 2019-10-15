import numpy as np
import tensorflow as tf

from .system import System

#################################################################################

class NematicLattice(System):
    """
    A 3-dimensional nematic lattice with vector units on each lattice site.
    """

    params_default = {
        "h" : 0.0,
        "J" : 1.0,
        "field" : np.zeros(3) # Orientation of the nematic field
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)

    def init_coords(self, N, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        shape = (N, N, N, 3)
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

    def draw_config(self, x):
        import matplotlib.pyplot as plt

        (fig, ax) = plt.subplots(1, figsize = (5, 5))

        N = x.shape[0]
        X, Y = np.meshgrid(range(N), range(N)) 
        
        plt.setp(ax.get_yticklabels(), visible = False)
        plt.setp(ax.get_xticklabels(), visible = False)     
        plt.pcolormesh(X, Y, x, cmap = plt.cm.RdBu)