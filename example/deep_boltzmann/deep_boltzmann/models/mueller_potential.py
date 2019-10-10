import numpy as np
import tensorflow as tf

class MuellerPotential(object):

    params_default = {'k' : 1.0,
                      'dim' : 2}


    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.dim = self.params['dim']

    def energy(self, x):
        """Muller potential

        Returns
        -------
        potential : {float, np.ndarray}
            Potential energy. Will be the same shape as the inputs, x and y.

        Reference
        ---------
        Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        value = 0
        for j in range(0, 4):
            value += self.AA[j] * np.exp(self.aa[j] * (x1 - self.XX[j])**2 +
                                         self.bb[j] * (x1 - self.XX[j]) * (x2 - self.YY[j]) +
                                         self.cc[j] * (x2 - self.YY[j])**2)
        # redundant variables
        if self.dim > 2:
            value += 0.5 * np.sum(x[:, 2:] ** 2, axis=1)

        return self.params['k'] * value

    def energy_tf(self, x):
        """Muller potential

        Returns
        -------
        potential : {float, np.ndarray}
            Potential energy. Will be the same shape as the inputs, x and y.

        Reference
        ---------
        Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        batchsize = tf.shape(x)[0]
        value = tf.zeros(batchsize)
        for j in range(0, 4):
            value += self.AA[j] * tf.exp(self.aa[j] * (x1 - self.XX[j])**2 +
                                         self.bb[j] * (x1 - self.XX[j]) * (x2 - self.YY[j]) +
                                         self.cc[j] * (x2 - self.YY[j])**2)
        # redundant variables
        if self.dim > 2:
            value += 0.5 * tf.reduce_sum(x[:, 2:] ** 2, axis=1)

        return self.params['k'] * value

    # def plot_dimer_energy(self, axis=None, temperature=1.0):
    #     """ Plots the dimer energy to the standard figure """
    #     x_grid = np.linspace(-3, 3, num=200)
    #     if self.dim == 1:
    #         X = x_grid[:, None]
    #     else:
    #         X = np.hstack([x_grid[:, None], np.zeros((x_grid.size, self.dim - 1))])
    #     energies = self.energy(X) / temperature
    #
    #     import matplotlib.pyplot as plt
    #     if axis is None:
    #         axis = plt.gca()
    #     #plt.figure(figsize=(5, 4))
    #     axis.plot(x_grid, energies, linewidth=3, color='black')
    #     axis.set_xlabel('x / a.u.')
    #     axis.set_ylabel('Energy / kT')
    #     axis.set_ylim(energies.min() - 2.0, energies[int(energies.size / 2)] + 2.0)
    #
    #     return x_grid, energies