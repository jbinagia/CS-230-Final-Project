import numpy as np
import tensorflow as tf

from utils import rotation_matrix
from .system import System

#################################################################################

class NematicLattice(System):
    """
    A 3-dimensional nematic lattice with vector units on each lattice site.
    """

    params_default = {
        "h" : 0.0,
        "J" : 1.0
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)
        self.field = np.array([0.0, 0.0, 1.0]) # Nematic field always points to z, coupling value `h` dictates contribution

    def init_coords(self, N, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        shape = (N, N, N, 3)
        x = 2.0*np.random.rand(*shape) - 1
        norms = np.linalg.norm(x, axis = 3)

        # Return normalized vectors of length 1
        return (x.reshape(np.prod(x.shape[:-1]), -1) / norms.reshape(norms.size, 1)).reshape(x.shape)

    def _neighbor_sum(self, x, i, j, k):
        N = x.shape[0]
        return x[(i+1)%N, j, k] + x[(i-1)%N, j, k] \
             + x[i, (j+1)%N, k] + x[i, (j-1)%N, k] \
             + x[i, j, (k+1)%N] + x[i, j, (k-1)%N]

    def energy(self, x):
        N = x.shape[0]
        en = 0.0    
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    s = x[i,j,k]
                    nb = self._neighbor_sum(x, i, j, k)
                    en += -0.25*self.params["J"]*np.dot(nb, s)
                    en += self.params["h"]*np.dot(self.field, s)
        return en

    def energy_idx(self, x, idx):
        N = x.shape[0]
        i, j, k = np.unravel_index(idx, (N, N, N))

        s = x[i,j,k]
        nb = self._neighbor_sum(x, i, j, k)
        en = -0.5*self.params["J"]*np.dot(nb, s)
        en += self.params["h"]*np.dot(self.field, s)

        return en

    def random_idx(self, x):
        return np.random.randint(np.prod(x.shape[:-1]))

    def step(self, x, **kwargs):
        theta = kwargs.get("theta", np.pi/4)
        M = rotation_matrix(np.random.rand(3), np.random.rand()*theta)
        return np.dot(M, x) # Make this work for other dimensions of x? Cannot pass whole array right now.

    def oprm(self, x):
        """Order parameter for a NematicLattice is average z-direction orientation."""
        return np.mean(x[:,:,:,2])

    #################################################################################

    def draw_config(self, x, figsize = (8, 8)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D 

        fig = plt.figure(figsize = figsize)
        ax = fig.gca(projection = '3d')

        N = len(x)
        gx, gy, gz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
        u, v, w = x.T

        # How to color 3D vectors (can alter cmap for variations)
        # Color by azimuthal angle
        c = np.arctan2(v, u)
        # Flatten and normalize
        c = (c.ravel() - c.min()) / c.ptp()
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.hsv(c)

        ax.quiver(gx, gy, gz, u, v, w, colors = c, length = 0.5)