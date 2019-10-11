import numpy as np

from .system import System
from .pbc import Box

#################################################################################

def _lj_pot(r, sig, eps):
    r6 = (sig / r) ** 6
    r12 = r6 * r6
    return 4.0 * eps * (r12 - r6)

class LJFluid(System):
    """
    A Lennard-Jones fluid in a periodic simulation box.
    """

    params_default = {
        "sig" : 1.0,
        "eps" : 1.0
    }

    def __init__(self, N, L, dim = 3, params = None, **kwargs):
        # Init super coordinate array
        shape = (N, dim)
        super().__init__(shape)

        self.N = N
        self.dim = dim

        # Add the useful params
        if params is None:
            params = self.__class__.params_default

        self.params = params
        self.sig = params["sig"]
        self.eps = params["eps"]

        # Construct a box
        self.box = Box([L] * dim)

        # Setup the coordinates
        if (self.N * (self.sig**self.dim)) > 0.9 * self.box.volume():
            raise ValueError("Particle volume exceeds simulation box size -- try increasing your box size.")
        self._init_coords(**kwargs)

        # Do initial energy methods
        self.calc_energy()

    def _init_coords(self, **kwargs):
        init_try = kwargs.get("init_try", 100)

        self.x[0, :] = self.box.random_position()
        for i in range(1, self.N):
            success = False
            for trial in range(init_try):
                new = self.box.random_position()

                dists = self.box.distance(new, self.x[:i])
                if not np.any(dists < 0.85 * self.sig):
                    self.x[i, :] = new
                    success = True
                    break

            if not success:
                raise RuntimeError("Coordinate initialization failure " \
                    + "-- likely too many particles, or too small of a box.")

    def calc_energy(self):
        # Raw calculation with O(N^2) scaling, slow
        energy = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                rij = self.box.distance(self.x[i], self.x[j])
                energy += _lj_pot(rij, self.sig, self.eps)
        self.energy = energy
        return energy

    def calc_energy_idx(self, idx):
        # For incremental MCMC updates, single particle, scales O(N)
        eidx = 0.0
        for i in range(self.N):
            if i == idx:
                continue
            else:
                rij = self.box.distance(self.x[idx], self.x[i])
                eidx += _lj_pot(rij, self.sig, self.eps)
        return eidx

    def displace(self, idx, **kwargs):
        step = kwargs.get("step", 0.5 * self.sig)
        return self.box.wrap(self.x[idx] + np.random.randn(self.dim) * step)

    def draw_config(self, axis = None, alpha = 0.7):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        # Prepare data
        X = self.x
        # Set up figure
        if axis is None:
            plt.figure(figsize=(5, 5))
            axis = plt.gca()

        d = self.box.lengths_half[0]
        axis.set_xlim((-d, d))
        axis.set_ylim((-d, d))
        # Draw box
        axis.add_patch(Rectangle((-d-self.params["sig"], -d-self.params["sig"]),
                                 2*d+2*self.params["sig"], 0.5*self.params["sig"], color='lightgrey', linewidth=0))
        axis.add_patch(Rectangle((-d-self.params["sig"], d+0.5*self.params["sig"]),
                                 2*d+2*self.params["sig"], 0.5*self.params["sig"], color='lightgrey', linewidth=0))
        axis.add_patch(Rectangle((-d-self.params["sig"], -d-self.params["sig"]),
                                 0.5*self.params["sig"], 2*d+2*self.params["sig"], color='lightgrey', linewidth=0))
        axis.add_patch(Rectangle((d+0.5*self.params["sig"], -d-self.params["sig"]),
                                 0.5*self.params["sig"], 2*d+2*self.params["sig"], color='lightgrey', linewidth=0))
        # Draw particles
        circles = []
        for x in X:
            circles.append(axis.add_patch(Circle(x - d, radius=0.5*self.params["sig"],
                                                 linewidth=2, edgecolor='black', facecolor='grey', alpha=alpha)))
        #plot(X[:, 0], X[:, 1], linewidth=0, marker='o', color='black')
        axis.set_xticks([])
        axis.set_yticks([])
        #return(fig, ax, circles)