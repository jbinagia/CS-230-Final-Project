import numpy as np

from .system import System
from .pbc import Box
from .potentials import lj_potential

#################################################################################

class LJFluid(System):
    """
    A Lennard-Jones fluid in a periodic simulation box.
    """

    params_default = {
        "sig" : 1.0,
        "eps" : 1.0
    }

    def __init__(self, nmon, L, dim = 3, params = None, **kwargs):
        # Init parent class
        shape = (nmon, dim)
        super().__init__(shape, params)

        self.nmon
        self.dim = dim
        self.box = Box([L] * dim)

        # Setup the coordinates
        if (self.nmon * (self.params["sig"]**self.dim)) > 0.9 * self.box.volume():
            raise ValueError("Particle volume exceeds simulation box size -- try increasing your box size.")

        # Initial coordinates and energy
        self._init_coords(**kwargs)
        self.calc_energy()

    def _init_coords(self, **kwargs):
        init_try = kwargs.get("init_try", 500)
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        self.x[0, :] = self.box.random_position()
        for i in range(1, self.nmon):
            success = False
            for trial in range(init_try):
                new = self.box.random_position()

                dists = self.box.distance(new, self.x[:i])
                if not np.any(dists < 0.85 * self.params["sig"]):
                    self.x[i, :] = new
                    success = True
                    break

            if not success:
                raise RuntimeError("Coordinate initialization failure " \
                    + "-- likely too many particles, or too small of a box.")

    def calc_energy(self):
        # Raw calculation with O(N^2) scaling, slow
        en = 0.0
        for i in range(self.nmon):
            for j in range(i+1, self.nmon):
                rij = self.box.distance(self.x[i], self.x[j])
                en += lj_potential(rij, self.params["sig"], self.params["eps"])

        self.energy = en
        return self.energy

    def calc_energy_idx(self, idx):
        # For incremental MCMC updates, single particle, scales O(N)
        eidx = 0.0
        for i in range(self.nmon):
            if i == idx:
                continue
            else:
                rij = self.box.distance(self.x[idx], self.x[i])
                eidx += lj_potential(rij, self.params["sig"], self.params["eps"])
                
        return eidx

    def displace(self, idx = None, **kwargs):
        step = kwargs.get("step", 0.5 * self.params["sig"])
        if idx:
            return self.box.wrap(self.x[idx] + np.random.randn(self.dim) * step)
        else:
            return self.box.wrap(self.x + np.random.randn(*self.shape) * step)

    def draw_config(self, alpha = 0.7):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        # Set up figure
        sig = self.params["sig"]

        if self.dim == 1:
            fig = plt.figure(figsize = (5, 0.5 * sig))
            axis = plt.gca()

            l = self.box.lengths[0]
            axis.set_xlim((-0.5*sig, l + 0.5*sig))
            axis.set_ylim((-0.5*sig, 0.5*sig))
        elif self.dim == 2:
            fig = plt.figure(figsize = (5, 5))
            axis = plt.gca()

            (lx, ly) = (self.box.lengths[0], self.box.lengths[1])
            axis.set_xlim((-0.5*sig, lx + 0.5*sig))
            axis.set_ylim((-0.5*sig, ly + 0.5*sig))
        else:
            raise RuntimeError("LJFluid drawing only available for 1D and 2D systems.")

        # axis.set_xticks([])
        # axis.set_yticks([])

        # Draw particles
        for x in self.x:
            if self.dim == 1:
                x = np.append(x, 0.0)
            axis.add_patch(
                Circle(x, radius = 0.5 * self.params["sig"],
                linewidth=2, edgecolor='black', facecolor='grey', alpha=alpha)
            )