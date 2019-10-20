import numpy as np

#################################################################################

class WangLandauSampler(object):

    def __init__(self, model, x0, bounds, nbins = 100, df = 0.5, tol = 0.2,
        temperature = 1.0, burnin = 0, mapper = None, **kwargs):
        """Metropolis Monte-Carlo simulation.

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x) and energy_idx(x)
        x0 : [array]
            Initial configuration
        temperatures : float or array
            Temperature. By default (1.0) the energy is interpreted in reduced units.
            When given an array, its length must correspond to nwalkers, then the walkers
            are simulated at different temperatures.
        burnin : int
            Number of burn-in steps that will not be saved
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.model = model
        self.temperature = temperature
        self.burnin = burnin
        self.kwargs = kwargs

        if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper

        # WL variables
        self.df = df
        self.tol = tol
        self.nbins = nbins

        doprm = 1/nbins if not kwargs.get("doprm") else kwargs.get("doprm")
        self.bounds = np.array([bounds[0] - doprm, bounds[1] + doprm])

        # Set initial coordinates and WL structures
        self.reset(x0)

    def _roughness(self):
        """Measure of flatness of histogram."""
        hm = np.mean(self.hist)
        rough = np.max(self.hist - hm) / hm
        return rough

    def _bin_id(self, oprm):
        bid = np.digitize(oprm, self.lims, right = False) - 1
        return bid

    def _proposal_step(self):
        # Proposal step
        self.idx_prop, self.x_prop = self.model.step(self.x, **self.kwargs)
        
        self.E_idx = self.model.energy_idx(self.x, self.idx_prop)
        self.E_idx_prop = self.model.energy_idx(self.x_prop, self.idx_prop)

        self.oprm_prop = self.model.oprm(self.x_prop)
        self.bid_prop = self._bin_id(self.oprm_prop)

    def _acceptance_step(self):
        # Only sample if we are within the bounded range, else reject w/o statistics?
        if 0 <= self.bid_prop < self.bins.size:

            # Acceptance step
            dE = self.E_idx_prop - self.E_idx
            dW = self.gn[self.bid_prop] - self.gn[self.bid]
            dE_WL = dE/self.temperature + dW

            acc = True if dE_WL < 0.0 else np.random.rand() < np.exp(-dE_WL)
            if acc:
                self.x = self.x_prop
                self.E = self.E + dE
                self.oprm = self.oprm_prop
                self.bid = self.bid_prop

            self.hist[self.bid] += 1      # Adjust the histogram for the new state
            self.gn[self.bid] += self.df  # Adjust the density of states by the log scale factor

        else:
            raise ValueError("Bin ID outside of histogram range.")

    def _reset_wl(self):
        self.epoch = 1

        self.lims = np.linspace(self.bounds[0], self.bounds[1], self.nbins + 2)
        self.bins = (self.lims[1:] + self.lims[:-1]) / 2
        self.hist = np.ones(self.bins.size)
        self.gn = -np.log(self.hist)

        self.oprm = self.model.oprm(self.x)
        self.bid = self._bin_id(self.oprm)

    def reset(self, x0):
        # Counters
        self.steps_ = []
        self.traj_ = []
        self.etraj_ = []
        self.otraj_ = []
        self.htraj_ = []
        self.gtraj_ = []

        # Initial configuration
        self.x = x0
        self.x = self.mapper.map(self.x)
        self.E = self.model.energy(self.x)

        # WL fields
        self._reset_wl()

        # Save first frame if no burnin
        if self.burnin == 0:
            self.steps_.append(0)
            self.traj_.append(self.x.copy())
            self.etraj_.append(self.E / self.model.num_sites(self.x))
            self.otraj_.append(self.oprm)
            self.htraj_.append(self.hist)
            self.gtraj_.append(self.gn)

    @property
    def steps(self):
        return np.array(self.steps_)

    @property
    def traj(self):
        return np.array(self.traj_)

    @property
    def etraj(self):
        return np.array(self.etraj_)

    @property
    def otraj(self):
        return np.array(self.otraj_)

    @property
    def htraj(self):
        return np.array(self.htraj_)

    @property
    def gtraj(self):
        return np.array(self.gtraj_)

    def run(self, nsteps, nwl = 100, stride = 1, verbose = False, **kwargs):
        if kwargs.get("seed"):
            np.random.seed(kwargs.get("seed"))

        step = 1
        if verbose > 0:
            print("Starting at WL-Epoch {}".format(self.epoch))

        while step <= nsteps:
            self._proposal_step()
            self._acceptance_step()

            # Store in trajectories
            if step > self.burnin and step % stride == 0:
                self.steps_.append(step)
                self.traj_.append(np.copy(self.x))
                self.etraj_.append(self.E / self.model.num_sites(self.x[0])) # Per-site energy trajectory
                self.otraj_.append(self.oprm)

            # Check for WL completeness
            if step % nwl == 0:
                if self._roughness() < self.tol:
                    self.epoch += 1

                    if verbose:
                        print("Entering WL-Epoch {:.0f} (step = {:.0f} / {:.0f})".format(self.epoch, step, nsteps))

                    self.df *= 0.5
                    self.gn = self.gn - np.max(self.gn)
                    self.htraj_.append(self.hist.copy())
                    self.gtraj_.append(-self.gn)
                    self.hist = np.ones(self.bins.size)

            step += 1
