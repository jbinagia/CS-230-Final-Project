"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# defining RealNVP network (https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb)
class RealNVP(nn.Module): # base class Module
    def __init__(self, nets, nett, mask, prior, system, input_dimension):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))]) # translation function (net)
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))]) # scaling function (net)
        # nn.ModuleList is basically just like a Python list, used to store a desired number of nn.Module’s.
        self.logp = 1.0 # initialize to 1
        self.system = system # class of what molecular system are we considering. E.g. Ising.
        self.orig_dimension = input_dimension # tuple describing original dim. of system. e.g. Ising Model with N = 8 would be (8,8)

    def g(self, z):
        log_R_zx, x = z.new_zeros(z.shape[0]), z

        for i in range(len(self.t)): # for each layer
            x_ = x*self.mask[i] # splitting features between channels.
                                # features selected here used to compute s(x) and f(x) but not updated themselves yet.
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_R_zx += torch.sum(s,-1)
        return x, log_R_zx

    def f(self, x):
        log_R_xz, z = x.new_zeros(x.shape[0]), x

        # new_zeros(size) returns a Tensor of size "size" filled with 0s
        for i in reversed(range(len(self.t))): # move backwards through layers
            z_ = self.mask[i] * z # tensor of size num samples x num features
            s = self.s[i](z_) * (1-self.mask[i]) # self.s[i] is the entire sequence of scaling operations
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_R_xz -= s.sum(dim=-1)
            # each pass through here applies all operations defined in nets() and all the ones defined in nett()
        # self.s[1](z_) is not the same as self.s[3](z_)
        self.log_R_xz = log_R_xz # save so we can reference it later
        return z, log_R_xz

    def forward(self, x):
        z, self.logp = self.f(x)
        return z

    def log_prob(self,x):
        z, logp = self.f(x) # z = f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample_n(batchSize) # was (batchSize,1), KH removed second dimension. This input is sample shape.
        logp = self.prior.log_prob(z)
        x, log_R_zx = self.g(z)
        return z.detach().numpy() , x.detach().numpy()

    def loss(self, batch, w_ml = 1.0, w_kl = 0.0, w_rc = 0.0):
        return w_ml*self.loss_ml(batch) + w_kl*self.loss_kl(batch) + w_rc*self.loss_rc(batch)

    def loss_ml(self, batch):
        z, log_R_xz = self.f(batch)
        self.energies = self.calculate_energy(batch)
        return self.expected_value(0.5*torch.norm(z,dim=1)**2 - log_R_xz)

    def loss_kl(self, batch_z,iter):
        x, log_R_zx = self.g(batch_z)

        self.energies = self.calculate_energy(x)
        # e_high = 10**4
        # if iter % 25 == 0:
        #     print("")
        #     print("Avg. E was: ", torch.norm(self.energies))
        # for energy in self.energies:
        #     if abs(energy) == float('inf'):
        #         print("energy overflow detected")
        #     elif energy > e_high:
        #         print("large energy detected")
        #         energy = e_high + torch.log(energy - e_high + 1.0)

        if iter % 25 == 0:
            print("Avg. position from batch is: ",torch.norm(batch_z)) # this should be constant
            print("Avg. g(z) is: ", torch.norm(x)) # this should change as g(z) changes
            print("Avg. E is (after regularizing): ", torch.norm(self.energies))
        return self.expected_value(self.energies - log_R_zx)

    def test_loss(self, batch_z, iter):
        self.energies = self.calculate_energy(self.g(batch_z)[0])
        if iter % 25 == 0:
            print("")
            print("Avg. position from batch is: ",torch.norm(batch_z)) # this should be constant
            print("Avg. g(z) is: ", torch.norm(self.g(batch_z)[0])) # this should change as g(z) changes
            print("Avg. E is: ", torch.norm(self.energies))
        return self.expected_value(self.energies)

    def loss_rc(self, batch):
        return 0.0

    def calculate_energy(self, batch):
        energies = batch.new_zeros(batch.shape[0])

        e_high = 10**4
        for i in range(batch.shape[0]): # for each x in the batch
            config = batch[i,:].reshape(self.orig_dimension) # reshape into correct form
            energies[i] = self.system.energy(config)
            if abs(energies[i]) == float('inf'):
                    print("energy overflow detected")
            elif energies[i] > e_high:
                    energies[i] = e_high + torch.log(energies[i] - e_high + 1.0)

        self.weights = torch.exp(-energies) # save Boltzmann weights
        return energies

    def expected_value(self, observable):
        return torch.dot(observable,self.weights)/torch.sum(self.weights)

def realnvp_loss_fn(z, model):
    """
    """

    return -(model.prior.log_prob(z) + model.logp).mean()
