import numpy as np
import tensorflow as tf

from .system import System

#################################################################################

class IsingModel(System):
    """
    A 2-dimensional Ising model grid.
    """

    params_default = {
      "h" : 0.0,
      "J" : 1.0
    }

    def __init__(self, N, params = None, **kwargs):
        # Init parent class
        shape = (N, N)
        super().__init__(shape, params)
        self.N = N

        # Initial coordinates and energy
        self._init_coords(**kwargs)
        self.calc_energy()

    def _init_coords(self, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))
        self.x = 2 * np.random.randint(0, 2, size = self.shape) - 1

    def calc_energy(self):
        en = 0.0
        for i in range(self.N):
            for j in range(self.N):
                sij = self.x[i,j]
                nb = self.x[(i+1)%self.N, j] + self.x[i,(j+1)%self.N] + self.x[(i-1)%self.N, j] + self.x[i,(j-1)%self.N]
                en += -0.25*self.params["J"]*nb*sij
                en += 0.5*self.params["h"]*sij 

        self.energy = en
        return self.energy

    def calc_energy_idx(self, idx):
        i = idx // self.N
        j = idx % self.N

        sij = self.x[i,j]
        nb = self.x[(i+1)%self.N, j] + self.x[i,(j+1)%self.N] + self.x[(i-1)%self.N, j] + self.x[i,(j-1)%self.N]
        eidx = -0.5*self.params["J"]*nb*sij 
        eidx += self.params["h"]*sij

        return eidx

    def displace(self, idx):
        i = idx // self.N
        j = idx % self.N
        return -1*self.x[i,j]

    def draw_config(self):
        import matplotlib.pyplot as plt

        (fig, ax) = plt.subplots(1, figsize = (5, 5))
        X, Y = np.meshgrid(range(self.N), range(self.N)) 
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)     
        plt.pcolormesh(X, Y, self.x, cmap=plt.cm.RdBu)