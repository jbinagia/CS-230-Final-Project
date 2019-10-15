__author__ = 'noe'

import numpy as np

#################################################################################

class MetropolisSampler(object):

    def __init__(self, model, x0, temperature = 1.0, burnin = 0, stride = 1, nwalkers = 1, 
        mapper = None, **kwargs):
        """Metropolis Monte-Carlo simulation.

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x) and energy_idx(x)
        x0 : [array]
            Initial configuration
        noise : float
            Noise intensity, standard deviation of Gaussian proposal step
        temperatures : float or array
            Temperature. By default (1.0) the energy is interpreted in reduced units.
            When given an array, its length must correspond to nwalkers, then the walkers
            are simulated at different temperatures.
        burnin : int
            Number of burn-in steps that will not be saved
        stride : int
            Every so many steps will be saved
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.model = model
        self.temperature = temperature
        self.burnin = burnin
        self.stride = stride
        self.nwalkers = nwalkers
        self.kwargs = kwargs

        if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper

        self.reset(x0)

    def _proposal_step(self):
        # Proposal step
        idx = self.model.random_idx(self.x)
        self.x_prop = self.model.displace(self.x, idx)
        self.x_prop = self.mapper.map(self.x_prop)

        
        self.E_prop = self.model.energy(self.x_prop)

    def _acceptance_step(self):
        # Acceptance step
        acc = -np.log(np.random.rand()) > (self.E_prop - self.E) / self.temperature
        self.x = np.where(acc[:, None], self.x_prop, self.x)
        self.E = np.where(acc, self.E_prop, self.E)

    def reset(self, x0):
        # Counters
        self.step = 0
        self.traj_ = []
        self.etraj_ = []

        # Initial configuration
        self.x = np.tile(x0, tuple([self.nwalkers]) + tuple([1]) * len(x0.shape) )
        self.x = self.mapper.map(self.x)
        self.E = np.array([self.model.energy(xi) for xi in self.x])

        # Save first frame if no burnin
        if self.burnin == 0:
            self.traj_.append(self.x)
            self.etraj_.append(self.E)

    @property
    def trajs(self):
        """Returns a list of trajectories, one trajectory for each walker."""
        T = np.array(self.traj_).astype(np.float32)
        return [T[:, i, :] for i in range(T.shape[1])]

    @property
    def traj(self):
        return self.trajs[0]

    @property
    def etrajs(self):
        """Returns a list of energy trajectories, one trajectory for each walker."""
        E = np.array(self.etraj_)
        return [E[:, i] for i in range(E.shape[1])]

    @property
    def etraj(self):
        return self.etrajs[0]

    def run(self, nsteps=1, verbose=0):
        for i in range(nsteps):
            self._proposal_step()
            self._acceptance_step()
            self.step += 1
            if verbose > 0 and i % verbose == 0:
                print('Step', i, '/', nsteps)
            if self.step > self.burnin and self.step % self.stride == 0:
                self.traj_.append(self.x)
                self.etraj_.append(self.E)


class ReplicaMetropolisSampler(object):

    def __init__(self, model, x0, temperatures, noise=0.1,
                 burnin=0, stride=1, mapper=None):
        if temperatures.size % 2 == 0:
            raise ValueError('Please use an odd number of temperatures.')
        self.temperatures = temperatures
        self.sampler = MetropolisSampler(model, x0, temperature=temperatures, noise=noise,
                                       burnin=burnin, stride=stride, nwalkers=temperatures.size, mapper=mapper)
        self.toggle=0

    @property
    def trajs(self):
        return self.sampler.trajs

    @property
    def etrajs(self):
        return self.sampler.etrajs

    def run(self, nepochs=1, nsteps_per_epoch=1, verbose=0):
        for i in range(nepochs):
            self.sampler.run(nsteps=nsteps_per_epoch)
            # exchange
            for k in range(self.toggle, self.temperatures.size-1, 2):
                c = -(self.sampler.E[k+1] - self.sampler.E[k]) * (1.0/self.temperatures[k+1] - 1.0/self.temperatures[k])
                acc = -np.log(np.random.rand()) > c
                if acc:
                    h = self.sampler.x[k].copy()
                    self.sampler.x[k] = self.sampler.x[k+1].copy()
                    self.sampler.x[k+1] = h
                    h = self.sampler.E[k]
                    self.sampler.E[k] = self.sampler.E[k+1]
                    self.sampler.E[k+1] = h
            self.toggle = 1 - self.toggle