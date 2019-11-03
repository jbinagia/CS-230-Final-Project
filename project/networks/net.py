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
            log_R_zx += s.sum(dim=1)
        return x, log_R_zx

    def f(self, x):
        log_R_xz, z = x.new_zeros(x.shape[0]), x
        # log_R_xz, z = x.new_zeros((x.shape[0],1)), x # if we want to keep the dimension

        # new_zeros(size) returns a Tensor of size "size" filled with 0s
        for i in reversed(range(len(self.t))): # move backwards through layers
            z_ = self.mask[i] * z # tensor of size num samples x num features
            s = self.s[i](z_) * (1-self.mask[i]) # self.s[i] is the entire sequence of scaling operations
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_R_xz -= s.sum(dim=1)
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
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x, log_R_zx = self.g(z)
        return x

    def loss(self, batch, w_ml = 1.0, w_kl = 0.0, w_rc = 0.0):
        return w_ml*self.loss_ml(batch) + w_kl*self.loss_kl(batch) + w_rc*self.loss_rc(batch)

    def loss_ml(self, batch):
        z, log_R_xz = self.f(batch)
        self.weights = self.calculate_weights(batch, z, log_R_xz)
        return self.expected_value(0.5*(torch.norm(z,dim=1) - log_R_xz), batch)

    def loss_kl(self, batch):
        # x, log_R_zx = self.g(z)
        log_R_zx = -self.log_R_xz
        return self.expected_value(self.system.energy(batch) - log_R_zx, batch)
        return 0.0

    def loss_rc(self, batch):
        return 0.0

    def calculate_weights(self, batch, z, log_R_xz):
        weights = batch.new_ones(batch.shape[0])
        for i in range(batch.shape[0]): # for each x in the batch
            log_prob_x = self.prior.log_prob(z[i:i+1,:]) + log_R_xz[i:i+1]
            #weights[i] = torch.exp(-self.system.energy(batch[i:i+1,:])-log_prob_x) # currently all weights are infinitely large
        return weights

    def expected_value(self, observable, batch):
        return torch.dot(observable,self.weights)/torch.sum(self.weights)

def realnvp_loss_fn(z, model):
    """
    """

    return -(model.prior.log_prob(z) + model.logp).mean()

# Define performance metrics related to our network architecture
def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
