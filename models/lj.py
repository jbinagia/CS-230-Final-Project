import numpy as np

from utils import lj_potential
from .system import System
from .pbc import Box

#################################################################################

class LJFluid(System):
    """
    A Lennard-Jones fluid in a periodic simulation box.
    """

    params_default = {
        "sig" : 1.0,
        "eps" : 1.0,
        "dim" : 3,
        "L"   : 5.0
    }

    def __init__(self, params = None, **kwargs):
        # Init parent class
        super().__init__(params, **kwargs)

        # Create an appropriate simulation box
        self.box = Box([self.params["L"]] * self.params["dim"])

    def init_coords(self, N, init_try = 100, **kwargs):
        # Check coordinate size
        if N * self._particle_volume() > 0.9 * self.box.volume():
            raise ValueError("Particle volume exceeds simulation box size -- try increasing your box size.")

        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        # Place while checking for overlap
        x = self.box.random_position()
        for i in range(1, N):
            success = False
            for trial in range(init_try):
                new = self.box.random_position()

                dists = self.box.distance(new, x)
                if not np.any(dists < 0.85 * self.params["sig"]):
                    x = np.vstack((x, new))
                    success = True
                    break

            if not success:
                raise RuntimeError("Coordinate initialization failure " \
                    + "-- likely too many particles, or too small of a box.")

        return x

    def energy(self, x):
        # Raw calculation with O(N^2) scaling, slow
        en = 0.0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                rij = self.box.distance(x[i], x[j])
                en += lj_potential(rij, self.params["sig"], self.params["eps"])
        return en

    def energy_idx(self, x, idx):
        # For incremental MCMC updates, single particle, scales O(N)
        en = 0.0
        for i in range(len(x)):
            if i == idx:
                continue
            else:
                rij = self.box.distance(x[idx], x[i])
                en += lj_potential(rij, self.params["sig"], self.params["eps"])
        return en

    def random_idx(self, x):
        return np.random.randint(x.shape[0])

    def displace(self, x, **kwargs):
        step = kwargs.get("step", 0.5 * self.params["sig"])
        return self.box.wrap(x + np.random.randn(*x.shape) * step)

    def _particle_volume(self):
        return self.params["sig"]**self.params["dim"]

    def oprm(self, x):
        """Order parameter for a LJFluid is particle volume fraction."""
        N = len(x)
        return N * self._particle_volume() / self.box.volume()

    #################################################################################

    def draw_config(self, x, alpha = 0.75, figsize = (8, 8)):
        if x.shape[-1] == 1:
            self._draw_config_1d(x, alpha, figsize)
        elif x.shape[-1] == 2:
            self._draw_config_2d(x, alpha, figsize)
        elif x.shape[-1] == 3:
            self._draw_config_3d(x, alpha, figsize)

    def _draw_config_1d(self, x, alpha, figsize = (8, 0.25)):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize = (figsize[0], 0.25))
        ax = plt.gca()

        sig = self.params["sig"]
        ax.set_xlim((-0.5*sig, self.box.lengths[0] + 0.5*sig))
        ax.set_ylim((-0.1, 0.1))
        ax.set_yticks([])

        # Plot as small points
        ax.scatter(x, np.zeros_like(x), c = "gray", edgecolors = 'black', alpha = alpha)


    def _draw_config_2d(self, x, alpha, figsize = (8, 8)):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig = plt.figure(figsize = figsize)
        ax = plt.gca()

        sig = self.params["sig"]
        (lx, ly) = (self.box.lengths[0], self.box.lengths[1])
        ax.set_xlim((-0.5*sig, lx + 0.5*sig))
        ax.set_ylim((-0.5*sig, ly + 0.5*sig))

        # Draw particles
        for c in x:
            ax.add_patch(Circle(c, radius = 0.5 * self.params["sig"],
                linewidth=2, edgecolor='black', facecolor='grey', alpha=alpha))

    def _draw_config_3d(self, x, alpha, figsize = (8, 8)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize = figsize)
        ax = Axes3D(fig)

        sig = self.params["sig"]
        x, y, z = x[:,0], x[:,1], x[:,2]
        ax.scatter(x, y, z, c = 'gray', edgecolors = 'black', s = 200 * sig)