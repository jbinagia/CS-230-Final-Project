import numpy as np

from .system import System
from .box import Box

from ..utils import lj_potential, distance_array

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

    def init_coords(self, N, init_try = 500, **kwargs):
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
        # Faster distance array computation, self-written (can find faster versions...)
        dists = distance_array(x, x, box = self.box)
        mu, nu = np.triu_indices_from(dists, k=1)
        return np.sum(lj_potential(dists[mu, nu], self.params["sig"], self.params["eps"]))

    def energy_idx(self, x, idx):
        # For incremental MCMC updates, single particle, scales O(N)
        dists = np.delete(self.box.distance(x[idx], x), idx)
        return np.sum(lj_potential(dists, self.params["sig"], self.params["eps"]))

    def step(self, x, **kwargs):
        delta = kwargs.get("delta", 0.5 * self.params["sig"])

        new = np.copy(x)
        idx = np.random.randint(x.shape[0])
        new[idx, :] = self.box.wrap(x[idx] + np.random.randn(x.shape[-1]) * delta)

        return idx, new

    def oprm(self, x):
        """Order parameter for a LJFluid is particle volume fraction."""
        N = len(x)
        return N * self._particle_volume() / self.box.volume()

    def num_sites(self, x):
        return x.shape[0]

    def _particle_volume(self):
        return self.params["sig"]**self.params["dim"]

    #################################################################################

    def draw_config(self, x, alpha = 0.75, figsize = (6, 6)):
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


    def _draw_config_2d(self, x, alpha, figsize = (6, 6)):
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

    def _draw_config_3d(self, x, alpha, figsize = (6, 6)):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize = figsize)
        ax = Axes3D(fig)

        sig = self.params["sig"]
        x, y, z = x[:,0], x[:,1], x[:,2]
        ax.scatter(x, y, z, c = 'gray', edgecolors = 'black', s = 200 * sig)