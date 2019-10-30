import numpy as np

#################################################################################

class Box(object):
    """
    A periodic simulation box in 3 dimensions.
    Implements methods for minimum image and position wrapping across the boundary.
    """

    def __init__(self, lengths):
        if type(lengths) not in [tuple, list, np.ndarray]:
            lengths = [lengths]

        self.dim = len(lengths)
        self.lengths = np.array(lengths).astype(float)
        self.lengths_half = self.lengths / 2.0

        self.h = np.diag(self.lengths)
        self.h_inv = np.linalg.inv(self.h)

    def volume(self):
        return np.prod(self.lengths)

    def wrap(self, x):
        nr = np.floor(x / self.lengths)
        x -= nr*self.lengths
        return x

    def min_image(self, x):
        nr = np.rint(x / self.lengths)
        x -= nr*self.lengths
        return x

    def distance(self, x1, x2):
        dx = self.min_image(x2 - x1)
        if len(dx.shape) > 1:
            return np.linalg.norm(dx, axis = 1)
        else:
            return np.linalg.norm(dx)

    def random_position(self):
        return np.dot(np.random.rand(self.dim), self.h)

    def center(self):
        return self.lengths_half