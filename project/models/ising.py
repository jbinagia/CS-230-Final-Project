import numpy as np

from .system import System
from .pbc import Box

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

    def energy(self, x):
        N = x.shape[0]

        en = 0.0    
        for i in range(N):
            for j in range(N):
                s = x[i,j]
                nb = self._neighbor_sum(x, i, j)
                en += -0.5*self.params["J"]*nb*s
                en += -self.params["h"]*s # Linear coupling to each lattice index

        return en

    def energy_idx(self, x, idx):
        N = x.shape[0]
        i, j = np.unravel_index(idx, (N, N))

        s = x[i,j]
        nb = self._neighbor_sum(x, i, j)
        en = -0.5*self.params["J"]*nb*s 
        en += -self.params["h"]*s

        return en

    def step(self, x, **kwargs):
        N = x.shape[0]
        i, j = np.random.randint(N, size = 2)
        idx = np.ravel_multi_index((i, j), (N, N))

        new = np.copy(x)
        new[i,j] = -1 * x[i,j]

        return idx, new

    def oprm(self, x):
        """Order parameter for IsingModel is the average magnetization."""
        return np.mean(x)

    def num_sites(self, x):
        N = x.shape[0]
        return N**2

    def _neighbor_sum(self, x, i, j):
        N = x.shape[0]
        return x[(i+1)%N, j] + x[(i-1)%N, j] \
             + x[i, (j+1)%N] + x[i, (j-1)%N]

    #################################################################################

    def draw_config(self, x, figsize = (6, 6)):
        import matplotlib.pyplot as plt

        (fig, ax) = plt.subplots(1, figsize = figsize)

        N = x.shape[0]
        X, Y = np.meshgrid(range(N+1), range(N+1)) 

        plt.setp(ax.get_yticklabels(), visible = False)
        plt.setp(ax.get_xticklabels(), visible = False)     
        plt.pcolormesh(X, Y, -1*x, cmap = plt.cm.RdBu)