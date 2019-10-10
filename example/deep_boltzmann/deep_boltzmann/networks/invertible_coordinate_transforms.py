import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
from deep_boltzmann.networks import IndexLayer


def log_det_jacobian(outputs, inputs):
    from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
    J = batch_jacobian(outputs, inputs, use_pfor=False)
    s = tf.svd(J, compute_uv=False)
    s = tf.abs(s) + 1e-6  # regularize
    return tf.reduce_sum(tf.log(s), axis=1, keepdims=True)


def pca(X0, keepdims=None):
    if keepdims is None:
        keepdims = X0.shape[1]
    # pca
    X0mean = X0.mean(axis=0)
    X0meanfree = X0 - X0mean
    C = np.dot(X0meanfree.T, X0meanfree) / (X0meanfree.shape[0] - 1.0)
    eigval, eigvec = np.linalg.eigh(C)
    # sort in descending order and keep only the wanted eigenpairs
    I = np.argsort(eigval)[::-1]
    I = I[:keepdims]
    eigval = eigval[I]
    std = np.sqrt(eigval)
    eigvec = eigvec[:, I]
    # whiten and unwhiten matrices
    X0mean = tf.constant(X0mean)
    Twhiten = tf.constant(eigvec.dot(np.diag(1.0 / std)))
    Tblacken = tf.constant(np.diag(std).dot(eigvec.T))
    return X0mean, Twhiten, Tblacken, std


class FixedWhiten(object):
    def __init__(self, X0, keepdims=None):
        """ Permutes dimensions

        Parameters:
        -----------
        X0 : array
            Initial Data on which PCA will be computed.
        keepdims : int or None
            Number of dimensions to keep. By default, all dimensions will be kept

        """
        if keepdims is None:
            keepdims = X0.shape[1]
        self.dim = X0.shape[1]
        self.keepdims = keepdims
        self.X0mean, self.Twhiten, self.Tblacken, self.std = pca(X0, keepdims=keepdims)
        if np.any(self.std <= 0):
            raise ValueError('Cannot construct whiten layer because trying to keep nonpositive eigenvalues.')
        self.jacobian_xz = -np.sum(np.log(self.std))

    @classmethod
    def from_dict(cls, D):
        dim = D['dim']
        keepdims = D['keepdims']
        X0mean = D['X0mean']
        Twhiten = D['Twhiten']
        Tblacken = D['Tblacken']
        std = D['std']
        c = cls(np.random.randn(2 * dim, dim), keepdims=keepdims)
        c.keepdims = keepdims
        c.X0mean = tf.constant(X0mean)
        c.Twhiten = tf.constant(Twhiten)
        c.Tblacken = tf.constant(Tblacken)
        c.std = std
        return c

    def to_dict(self):
        D = {}
        D['dim'] = self.dim
        D['keepdims'] = self.keepdims
        D['X0mean'] = keras.backend.eval(self.X0mean)
        D['Twhiten'] = keras.backend.eval(self.Twhiten)
        D['Tblacken'] = keras.backend.eval(self.Tblacken)
        D['std'] = self.std
        return D

    def connect_xz(self, x):
        # Whiten
        self.output_z = keras.layers.Lambda(lambda x: tf.matmul(x - self.X0mean, self.Twhiten))(x)
        if self.keepdims < self.dim:
            junk_dims = self.dim - self.keepdims
            self.output_z = keras.layers.Lambda(lambda z: tf.concat([z, tf.random_normal([tf.shape(z)[0], junk_dims], stddev=1.0)], 1))(self.output_z)
        # Jacobian
        self.log_det_xz = keras.layers.Lambda(lambda x: self.jacobian_xz * keras.backend.ones((tf.shape(x)[0], 1)))(x)
        return self.output_z

    def connect_zx(self, z):
        # if we have reduced the dimension, we ignore the last dimensions from the z-direction.
        if self.keepdims < self.dim:
            z = IndexLayer(np.arange(0, self.keepdims))(z)
        self.output_x = keras.layers.Lambda(lambda z: tf.matmul(z, self.Tblacken) + self.X0mean)(z)
        # Jacobian
        self.log_det_zx = keras.layers.Lambda(lambda z: -self.jacobian_xz * keras.backend.ones((tf.shape(z)[0], 1)))(z)
        return self.output_x

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_zx



def xyz2ic_np(x, Z_indices, torsion_cut=None):
    """ Computes internal coordinates from Cartesian coordinates

    Parameters
    ----------
    x : array
        Catesian coordinates
    Z_indices : array
        Internal coordinate index definition. Use -1 to switch off internal coordinates
        when the coordinate system is not fixed.
    mm : energy model
        Molecular model
    torsion_cut : None or array
        If given, defines at which angle to cut the torsions.

    """
    from deep_boltzmann.models.MM import dist, angle, torsion
    global_ic = (Z_indices.min() < 0)
    if global_ic:
        bond_indices = Z_indices[1:, :2]
        angle_indices = Z_indices[2:, :3]
        torsion_indices = Z_indices[3:, :4]
    else:
        bond_indices = Z_indices[:, :2]
        angle_indices = Z_indices[:, :3]
        torsion_indices = Z_indices[:, :4]
    atom_indices = np.arange(int(3*(np.max(Z_indices)+1))).reshape((-1, 3))
    xbonds = dist(x[:, atom_indices[bond_indices[:, 0]]],
                  x[:, atom_indices[bond_indices[:, 1]]])
    xangles = angle(x[:, atom_indices[angle_indices[:, 0]]],
                    x[:, atom_indices[angle_indices[:, 1]]],
                    x[:, atom_indices[angle_indices[:, 2]]])
    xtorsions = torsion(x[:, atom_indices[torsion_indices[:, 0]]],
                        x[:, atom_indices[torsion_indices[:, 1]]],
                        x[:, atom_indices[torsion_indices[:, 2]]],
                        x[:, atom_indices[torsion_indices[:, 3]]])
    if torsion_cut is not None:
        xtorsions = np.where(xtorsions < torsion_cut, xtorsions+360, xtorsions)
    # Order ic's by atom
    if global_ic:
        iclist = [xbonds[:, 0:2], xangles[:, 0:1]]
        for i in range(Z_indices.shape[0]-3):
            iclist += [xbonds[:, i+2:i+3], xangles[:, i+1:i+2], xtorsions[:, i:i+1]]
    else:
        iclist = []
        for i in range(Z_indices.shape[0]):
            iclist += [xbonds[:, i:i+1], xangles[:, i:i+1], xtorsions[:, i:i+1]]
    ics = np.concatenate(iclist, axis=-1)
    return ics


def xyz2ic_tf(x, Z_indices, torsion_cut=None):
    """ Computes internal coordinates from Cartesian coordinates

    Parameters
    ----------
    x : array
        Catesian coordinates
    Z_indices : array
        Internal coordinate index definition. Use -1 to switch off internal coordinates
        when the coordinate system is not fixed.
    mm : energy model
        Molecular model
    torsion_cut : None or array
        If given, defines at which angle to cut the torsions.

    """
    from deep_boltzmann.models.MM import dist_tf, angle_tf, torsion_tf
    global_ic = (Z_indices.min() < 0)
    if global_ic:
        bond_indices = Z_indices[1:, :2]
        angle_indices = Z_indices[2:, :3]
        torsion_indices = Z_indices[3:, :4]
    else:
        bond_indices = Z_indices[:, :2]
        angle_indices = Z_indices[:, :3]
        torsion_indices = Z_indices[:, :4]
    atom_indices = np.arange(int(3*(np.max(Z_indices)+1))).reshape((-1, 3))
    xbonds = dist_tf(tf.gather(x, atom_indices[bond_indices[:, 0]], axis=1),
                     tf.gather(x, atom_indices[bond_indices[:, 1]], axis=1))
    xangles = angle_tf(tf.gather(x, atom_indices[angle_indices[:, 0]], axis=1),
                       tf.gather(x, atom_indices[angle_indices[:, 1]], axis=1),
                       tf.gather(x, atom_indices[angle_indices[:, 2]], axis=1))
    xtorsions = torsion_tf(tf.gather(x, atom_indices[torsion_indices[:, 0]], axis=1),
                           tf.gather(x, atom_indices[torsion_indices[:, 1]], axis=1),
                           tf.gather(x, atom_indices[torsion_indices[:, 2]], axis=1),
                           tf.gather(x, atom_indices[torsion_indices[:, 3]], axis=1))
    if torsion_cut is not None:
        xtorsions = tf.where(xtorsions < torsion_cut, xtorsions+360, xtorsions)
    # Order ic's by atom
    if global_ic:
        iclist = [xbonds[:, 0:2], xangles[:, 0:1]]
        for i in range(Z_indices.shape[0]-3):
            iclist += [xbonds[:, i+2:i+3], xangles[:, i+1:i+2], xtorsions[:, i:i+1]]
    else:
        iclist = []
        for i in range(Z_indices.shape[0]):
            iclist += [xbonds[:, i:i+1], xangles[:, i:i+1], xtorsions[:, i:i+1]]
    ics = tf.concat(iclist, axis=-1)
    return ics

def bestcut(torsion):
    cuts = np.linspace(-180, 180, 37)[:-1]
    stds = []
    for cut in cuts:
        torsion_cut = np.where(torsion < cut, torsion+360, torsion)
        stds.append(np.std(torsion_cut))
    stds = np.array(stds)
    stdmin = stds.min()
    minindices = np.where(stds == stdmin)[0]
    return cuts[minindices[int(0.5*minindices.size)]]

def icmoments(Z_indices, X0=None, torsion_cut=None):
    global_ic = (Z_indices.min() < 0)
    if global_ic:
        dim = 3*Z_indices.shape[0] - 6
        ntorsions = Z_indices.shape[0]-3
    else:
        dim = 3*Z_indices.shape[0]
        ntorsions = Z_indices.shape[0]

    if X0 is not None:
        ics = xyz2ic_np(X0, Z_indices)
        if global_ic:
            torsions = ics[:, 5::3]
        else:
            torsions = ics[:, 2::3]
        if torsion_cut is None:
            torsion_cut = np.array([bestcut(torsions[:, i]) for i in range(ntorsions)])
        # apply torsion cut
        torsion_cut_row = np.array([torsion_cut])
        torsions = np.where(torsions < torsion_cut_row, torsions+360, torsions)
        # write torsions back to ics
        if global_ic:
            ics[:, 5::3] = torsions
        else:
            ics[:, 2::3] = torsions
        means = np.mean(ics, axis=0)
        stds = np.sqrt(np.mean((ics-means) ** 2, axis=0))
    else:
        torsion_cut = -180 * np.ones((1, ntorsions))
        means = np.zeros((1, dim-6))
        stds = np.ones((1, dim-6))
    return means, stds, torsion_cut

def ic2xyz(p1, p2, p3, d14, a124, t1234):
    # convert angles to radians
    a124 = a124 * np.pi/180.0
    t1234 = t1234 * np.pi/180.0
    v1 = p1 - p2
    v2 = p1 - p3

    n = tf.cross(v1, v2)
    nn = tf.cross(v1, n)
    n /= tf.norm(n, axis=1, keepdims=True)
    nn /= tf.norm(nn, axis=1, keepdims=True)

    n *= -tf.sin(t1234)
    nn *= tf.cos(t1234)

    v3 = n + nn
    v3 /= tf.norm(v3, axis=1, keepdims=True)
    v3 *= d14 * tf.sin(a124)

    v1 /= tf.norm(v1, axis=1, keepdims=True)
    v1 *= d14 * tf.cos(a124)

    position = p1 + v3 - v1

    return position

def ic2xy0(p1, p2, d14, a124):
    #t1234 = tf.Variable(np.array([[90.0 * np.pi / 180.0]], dtype=np.float32))
    t1234 = tf.Variable(np.array([[90.0]], dtype=np.float32))
    p3 = tf.Variable(np.array([[0, 1, 0]], dtype=np.float32))
    return ic2xyz(p1, p2, p3, d14, a124, t1234)

def ics2xyz_global(ics, Z_indices):
    """ For systems exclusively described in internal coordinates: convert global Z matrix to Cartesian """
    batchsize = tf.shape(ics)[0]
    index2zorder = np.argsort(Z_indices[:, 0])
    # Fix coordinate system by placing first three atoms
    xyz = []
    # first atom at 0,0,0
    xyz.append(tf.zeros((batchsize, 3)))
    # second atom at 0,0,d
    xyz.append(tf.concat([tf.zeros((batchsize, 2)), ics[:, 0:1]], axis=-1))
    # third atom at x,0,z
    xyz.append(ic2xy0(xyz[index2zorder[Z_indices[2, 1]]],
                      xyz[index2zorder[Z_indices[2, 2]]],
                      ics[:, 1:2], ics[:, 2:3]))
    # fill in the rest
    ics2xyz_local(ics[:, 3:], Z_indices[3:], index2zorder, xyz)

    # reorganize indexes
    xyz = [xyz[i] for i in index2zorder]
    return tf.concat(xyz, axis=1)

def ics2xyz_local(ics, Z_indices, index2zorder, xyz):
    """ For systems exclusively described in internal coordinates: convert global Z matrix to Cartesian

    Parameters
    ----------
    ics : array (batchsize x dim)
        IC matrix flattened by atom to place (bond1, angle1, torsion1, bond2, angle2, torsion2, ...)

    """
    for i in range(Z_indices.shape[0]):
        xyz.append(ic2xyz(xyz[index2zorder[Z_indices[i, 1]]],
                          xyz[index2zorder[Z_indices[i, 2]]],
                          xyz[index2zorder[Z_indices[i, 3]]],
                          ics[:, 3*i:3*i+1], ics[:, 3*i+1:3*i+2], ics[:, 3*i+2:3*i+3]))

# def ics2xyz_local_log_det_jac(ics, Z_indices, index2zorder, xyz):
#
#     batchsize = tf.shape(ics)[0]
#
#     log_det_jac = tf.zeros((batchsize,))
#
#     for i in range(Z_indices.shape[0]):
#         args = tf.concat([
#             ics[:, 3*i:3*i+1],
#             ics[:, 3*i+1:3*i+2],
#             ics[:, 3*i+2:3*i+3]
#         ], axis=-1)
#         xyz.append(ic2xyz(xyz[index2zorder[Z_indices[i, 1]]],
#                           xyz[index2zorder[Z_indices[i, 2]]],
#                           xyz[index2zorder[Z_indices[i, 3]]],
#                           args[..., 0:1], args[..., 1:2], args[..., 2:3]))
#         log_det_jac += tf.linalg.slogdet(batch_jacobian(xyz[-1], args))[-1]
#
#     return log_det_jac
def ics2xyz_local_log_det_jac(ics, Z_indices, index2zorder, xyz):
    batchsize = tf.shape(ics)[0]
    log_det_jac = tf.zeros((batchsize,))

    for i in range(Z_indices.shape[0]):
        args = ics[:, 3*i:3*i+3]
        xyz.append(ic2xyz(xyz[index2zorder[Z_indices[i, 1]]],
                          xyz[index2zorder[Z_indices[i, 2]]],
                          xyz[index2zorder[Z_indices[i, 3]]],
                          args[:, 0:1], args[:, 1:2], args[:, 2:3]))
        log_det_jac += tf.linalg.slogdet(batch_jacobian(xyz[-1], args))[-1]

    return log_det_jac

def log_det_jac_lists(ys, xs):
    from tensorflow.python.ops import gradients as gradient_ops

    batch_dim = xs[0].shape[0]
    output_dim = ys[0].shape[-1]

    jacs = []
    for y, x in zip(ys, xs):
        cols = []
        for i in range(output_dim):
            cols.append(gradient_ops.gradients(y[:, i], x)[0])
        jac = tf.stack(cols, axis=-1)
        jacs.append(jac)

    log_det = tf.linalg.slogdet(jacs)[-1]
    log_det = tf.reduce_sum(log_det, axis=0)

    return log_det

def ics2xyz_local_log_det_jac_lists(ics, Z_indices, index2zorder, xyz):

    batchsize = tf.shape(ics)[0]

    log_det_jac = tf.zeros((batchsize,))
    all_args = []
    all_outputs = []

    for i in range(Z_indices.shape[0]):
        all_args.append(tf.concat([
            ics[:, 3*i:3*i+1],
            ics[:, 3*i+1:3*i+2],
            ics[:, 3*i+2:3*i+3]
        ], axis=-1))
        xyz.append(ic2xyz(xyz[index2zorder[Z_indices[i, 1]]],
                          xyz[index2zorder[Z_indices[i, 2]]],
                          xyz[index2zorder[Z_indices[i, 3]]],
                          all_args[-1][..., 0:1], all_args[-1][..., 1:2], all_args[-1][..., 2:3]))
        all_outputs.append(xyz[-1])

    log_det_jac = log_det_jac_lists(all_outputs, all_args)

    return log_det_jac


def decompose_Z_indices(cart_indices, Z_indices):
    known_indices = cart_indices
    Z_placed = np.zeros(Z_indices.shape[0])
    Z_indices_decomposed = []
    while np.count_nonzero(Z_placed) < Z_indices.shape[0]:
        Z_indices_cur = []
        for i in range(Z_indices.shape[0]):
            if not Z_placed[i] and np.all([Z_indices[i, j] in known_indices for j in range(1, 4)]):
                Z_indices_cur.append(Z_indices[i])
                Z_placed[i] = 1
        Z_indices_cur = np.array(Z_indices_cur)
        known_indices = np.concatenate([known_indices, Z_indices_cur[:, 0]])
        Z_indices_decomposed.append(Z_indices_cur)

    index2order = np.concatenate([cart_indices] + [Z[:, 0] for Z in Z_indices_decomposed])

    return Z_indices_decomposed, index2order

def ics2xyz_local_log_det_jac_batchexpand(ics, Z_indices, index2zorder, xyz, eps=1e-10):
    """ For systems exclusively described in internal coordinates: convert global Z matrix to Cartesian

    Parameters
    ----------
    ics : array (batchsize x dim)
        IC matrix flattened by atom to place (bond1, angle1, torsion1, bond2, angle2, torsion2, ...)

    """
    from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
    from deep_boltzmann.networks.invertible_coordinate_transforms import ic2xyz
    batchsize = tf.shape(ics)[0]
    natoms_to_place = Z_indices.shape[0]

    # reshape atoms into the batch
    p1s = tf.reshape(tf.gather(xyz, index2zorder[Z_indices[:, 1]], axis=1), (batchsize*natoms_to_place, 3))
    p2s = tf.reshape(tf.gather(xyz, index2zorder[Z_indices[:, 2]], axis=1), (batchsize*natoms_to_place, 3))
    p3s = tf.reshape(tf.gather(xyz, index2zorder[Z_indices[:, 3]], axis=1), (batchsize*natoms_to_place, 3))
    ics_ = tf.reshape(ics, (batchsize*natoms_to_place, 3))

    # operation to differentiate: compute new xyz's given distances, angles, torsions
    newpos_batchexpand = ic2xyz(p1s, p2s, p3s, ics_[:, 0:1], ics_[:, 1:2], ics_[:, 2:3])
    newpos = tf.reshape(newpos_batchexpand, (batchsize, natoms_to_place, 3))

    # compute derivatives
    log_det_jac_batchexpand = tf.linalg.slogdet(batch_jacobian(newpos_batchexpand, ics_))[-1]

    # reshape atoms again out of batch and sum over the log det jacobians
    log_det_jac = tf.reshape(log_det_jac_batchexpand, (batchsize, natoms_to_place))
    log_det_jac = tf.reduce_sum(log_det_jac, axis=1)

    return newpos, log_det_jac


def ics2xyz_local_log_det_jac_decomposed(all_ics, all_Z_indices, cartesian_xyz, index2order, eps=1e-10):
    """
    Parameters
    ----------
    all_ics : Tensor (batchsize, 3*nICatoms)
        Tensor with all internal coordinates to be placed, in the order as they are placed in all_Z_indices
    all_Z_indices : list of Z index arrays.
        All atoms in one array are placed independently given the atoms that have been placed before
    cartesian_xyz : Tensor (batchsize, nCartAtoms, 3)
        Start here with the positions of all Cartesian atoms
    index2order : array
        map from atom index to the order of placement. The order of placement is first all Cartesian atoms
        and then in the order of np.vstack(all_Z_indices)[:, 0]

    """
    batchsize = tf.shape(all_ics)[0]
    log_det_jac_tot = tf.zeros((batchsize), )
    xyz = cartesian_xyz
    istart = 0
    for Z_indices in all_Z_indices:
        ics = all_ics[:, 3*istart:3*(istart+Z_indices.shape[0])]
        newpos, log_det_jac = ics2xyz_local_log_det_jac_batchexpand(ics, Z_indices, index2order, xyz, eps=eps)
        xyz = tf.concat([xyz, newpos], axis=1)
        log_det_jac_tot += log_det_jac
        istart += Z_indices.shape[0]
    return xyz, log_det_jac_tot


def ics2xyz_global_log_det_jac(ics, Z_indices, global_transform=True):
    batchsize = tf.shape(ics)[0]

    index2zorder = np.argsort(Z_indices[:, 0])

    xyz = []

    log_det_jac = tf.zeros((batchsize,))

    if global_transform:
        # first atom at 0,0,0
        xyz.append(tf.zeros((batchsize, 3)))

        # second atom at 0,0,d
        args = tf.reshape(ics[:, 0:1], (batchsize, 1))
        xyz.append(tf.concat([tf.zeros((batchsize, 2)), args], axis=-1))
        z = xyz[-1][:, -1:]

        log_det_jac += tf.linalg.slogdet(batch_jacobian(z, args))[-1]

        # third atom at x,0,z
        args = tf.concat([ics[:, 1:2], ics[:, 2:3]], axis=-1)
        xyz.append(ic2xy0(xyz[index2zorder[Z_indices[2, 1]]],
                          xyz[index2zorder[Z_indices[2, 2]]],
                          args[..., 0:1], args[..., 1:2]))
        xz = tf.stack([xyz[-1][:, 0], xyz[-1][:, 2]], axis=-1)
        #  + 1e-6*tf.eye(2, num_columns=2, batch_shape=(1,)
        log_det_jac += tf.linalg.slogdet(batch_jacobian(xz, args))[-1]

    # other atoms
    log_det_jac += ics2xyz_local_log_det_jac(
        ics[:, 3:], Z_indices[3:], index2zorder, xyz)

    return log_det_jac


def xyz2ic_log_det_jac(x, Z_indices, eps=1e-10):

    from deep_boltzmann.models.MM import dist_tf, angle_tf, torsion_tf

    batchsize = tf.shape(x)[0]

    atom_indices = np.arange(int(3*(np.max(Z_indices)+1))).reshape((-1, 3))

    log_det_jac = tf.zeros((batchsize,))

    global_transform = (Z_indices.min() < 0)
    if global_transform:
        start_rest = 3  # remaining atoms start in row 3

        # 1. bond (input: z axis)
        reference_atom = tf.gather(x, atom_indices[Z_indices[1, 0]], axis=1)
        other_atom = tf.gather(x, atom_indices[Z_indices[1, 1]], axis=1)

        x_ = reference_atom[:, 0]
        y_ = reference_atom[:, 1]
        z_ = reference_atom[:, 2]

        arg = tf.expand_dims(z_, axis=1)
        reference_atom = tf.stack([x_, y_, arg[:, 0]], axis=-1)

        reference_atom = tf.expand_dims(reference_atom, axis=1)
        other_atom = tf.expand_dims(other_atom, axis=1)

        bond = dist_tf(
            reference_atom,
            other_atom
        )

        out = bond
        jac = batch_jacobian(out, arg) + eps * tf.eye(3, batch_shape=(1,))
        log_det_jac += tf.linalg.slogdet(jac)[-1]

        # 2. bond/angle (input: x/z axes)
        reference_atom = tf.gather(x, atom_indices[Z_indices[2, 0]], axis=1)
        other_atom_1 = tf.gather(x, atom_indices[Z_indices[2, 1]], axis=1)
        other_atom_2 = tf.gather(x, atom_indices[Z_indices[2, 2]], axis=1)

        x_ = reference_atom[:, 0]
        y_ = reference_atom[:, 1]
        z_ = reference_atom[:, 2]

        arg = tf.stack([x_, z_], axis=-1)
        reference_atom = tf.stack([arg[:, 0], y_, arg[:, 1]], axis=-1)

        reference_atom = tf.expand_dims(reference_atom, axis=1)
        other_atom_1 = tf.expand_dims(other_atom_1, axis=1)
        other_atom_2 = tf.expand_dims(other_atom_2, axis=1)

        bond = dist_tf(
            reference_atom,
            other_atom_1
        )
        angle = angle_tf(
            reference_atom,
            other_atom_1,
            other_atom_2
        )
        out = tf.stack([bond, angle], axis=-1)
        jac = batch_jacobian(out, arg) + eps * tf.eye(3, batch_shape=(1,))

        log_det_jac_ = tf.linalg.slogdet(jac)[-1]
        log_det_jac_ = tf.reshape(log_det_jac_, [batchsize, -1])
        log_det_jac_ = tf.reduce_sum(log_det_jac_, axis=-1)
        log_det_jac += log_det_jac_
    else:
        start_rest = 0  # remaining atoms start now

    # 3. everything together
    reference_atoms = tf.gather(x, atom_indices[Z_indices[start_rest:, 0]], axis=1)
    other_atoms_1 = tf.gather(x, atom_indices[Z_indices[start_rest:, 1]], axis=1)
    other_atoms_2 = tf.gather(x, atom_indices[Z_indices[start_rest:, 2]], axis=1)
    other_atoms_3 = tf.gather(x, atom_indices[Z_indices[start_rest:, 3]], axis=1)

    arg = tf.reshape(reference_atoms, [-1, 3])
    reference_atoms = tf.reshape(arg, [batchsize, -1, 3])

    bond = dist_tf(
        reference_atoms,
        other_atoms_1
    )
    angle = angle_tf(
        reference_atoms,
        other_atoms_1,
        other_atoms_2
    )
    torsion = torsion_tf(
        reference_atoms,
        other_atoms_1,
        other_atoms_2,
        other_atoms_3
    )
    out = tf.stack([bond, angle, torsion], axis=-1)
    out = tf.reshape(out, [-1, 3])
    jac = batch_jacobian(out, arg, use_pfor=False) # + eps * tf.eye(3, batch_shape=(1,)

    log_det_jac_ = tf.linalg.slogdet(jac)[-1]
    log_det_jac_ = tf.reshape(log_det_jac_, [batchsize, -1])
    log_det_jac_ = tf.reduce_sum(log_det_jac_, axis=-1)
    log_det_jac += log_det_jac_

    return log_det_jac


class InternalCoordinatesTransformation(object):
    """ Conversion between internal and Cartesian coordinates """

    def __init__(self, Z_indices, Xnorm=None, torsion_cut=None):
        self.dim = Z_indices.shape[0] * 3
        self.Z_indices = Z_indices

        # Compute IC moments for normalization
        self.ic_means, self.ic_stds, self.torsion_cut = icmoments(Z_indices, X0=Xnorm, torsion_cut=torsion_cut)


    @classmethod
    def from_dict(cls, D):
        ic_means = D['ic_means']
        ic_stds = D['ic_stds']
        torsion_cut = D['torsion_cut']
        Z_indices = D['Z_indices']
        c = cls(Z_indices)
        c.ic_means = ic_means
        c.ic_stds = ic_stds
        c.torsion_cut = torsion_cut
        return c

    def to_dict(self):
        D = {}
        D['ic_means'] = self.ic_means
        D['ic_stds'] = self.ic_stds
        D['torsion_cut'] = self.torsion_cut
        D['Z_indices'] = self.Z_indices
        return D

    def x2z(self, x):
        # compute and normalize internal coordinates
        z_ics = xyz2ic_tf(x, self.Z_indices, torsion_cut=self.torsion_cut)
        z_ics_norm = (z_ics - self.ic_means) / self.ic_stds

        return z_ics_norm

    def z2x(self, z):
        # split off Z block
        z_ics_unnorm = z * self.ic_stds + self.ic_means
        # reconstruct remaining atoms using ICs
        x = ics2xyz_global(z_ics_unnorm, self.Z_indices)

        return x

    def x2z_jacobian(self, x):
        log_det_jac = xyz2ic_log_det_jac(x, self.Z_indices)
        log_det_jac = tf.reshape(log_det_jac, (-1, 1))
        # log_det_jac -= tf.reduce_sum(tf.log(self.ic_stds))
        return log_det_jac

    def z2x_jacobian(self, z):
        log_det_jac = ics2xyz_global_log_det_jac(z, self.Z_indices)
        log_det_jac = tf.reshape(log_det_jac, (-1, 1))
        # log_det_jac += tf.reduce_sum(tf.log(self.ic_stds))
        return log_det_jac

    def connect_xz(self, x):
        self.input_x = x
        self.output_z_only = keras.layers.Lambda(lambda x: self.x2z(x))(x)
        junk_dims = 6
        self.output_z = keras.layers.Lambda(
                lambda z: tf.concat([z, 0. * tf.random_normal([tf.shape(z)[0], junk_dims], stddev=1.)], 1))(self.output_z_only)

        # self.log_det_xz = keras.layers.Lambda(lambda x: log_det_jacobian(self.x2z(x), x))(self.input_x)
        self.log_det_xz = keras.layers.Lambda(lambda x: self.x2z_jacobian(x))(self.input_x)

        return self.output_z

    def connect_zx(self, z):
        self.input_z = z
        z = IndexLayer(np.arange(0, self.dim-6))(z)
        self.output_x = keras.layers.Lambda(lambda z: self.z2x(z)[0])(z)
        self.angle_loss = keras.layers.Lambda(lambda z: self.z2x(z)[1])(z)

        # self.log_det_zx = keras.layers.Lambda(lambda z: log_det_jacobian(self.z2x(z), z))(z)
        self.log_det_zx = keras.layers.Lambda(lambda z: self.z2x_jacobian(z))(z)

        return self.output_x

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_zx

class MixedCoordinatesTransformation(InternalCoordinatesTransformation):
    """ Conversion between Cartesian coordinates and whitened Cartesian / whitened internal coordinates """

    def __init__(self, cart_atom_indices, Z_indices_no_order, X0=None, X0ic=None, remove_dof=6, torsion_cut=None,
                 jacobian_regularizer=1e-10):
        """
        Parameters
        ----------
        mm : energy model
            Molecular Model
        cart_atom_indices : array
            Indices of atoms treated as Cartesian, will be whitened with PCA
        ic_atom_indices : list of arrays
            Indices of atoms for which internal coordinates will be computed. Each array defines the Z matrix
            for that IC group.
        X0 : array or None
            Initial coordinates to compute whitening transformations on.
        remove_dof : int
            Number of degrees of freedom to remove from PCA whitening (default is 6 for translation+rotation in 3D)

        """
        self.cart_atom_indices = np.array(cart_atom_indices)
        self.cart_indices = np.concatenate([[i*3, i*3+1, i*3+2] for i in cart_atom_indices])
        self.batchwise_Z_indices, _ = decompose_Z_indices(self.cart_atom_indices, Z_indices_no_order)
        self.Z_indices = np.vstack(self.batchwise_Z_indices)
        self.dim = 3*(self.cart_atom_indices.size + self.Z_indices.shape[0])
        self.atom_order = np.concatenate([cart_atom_indices, self.Z_indices[:, 0]])
        self.index2order = np.argsort(self.atom_order)
        self.remove_dof = remove_dof
        self.jacobian_regularizer = jacobian_regularizer

        if X0 is None:
            raise ValueError('Need to specify X0')
        if X0ic is None:
            X0ic = X0

        # Compute PCA transformation on initial data
        self.cart_X0mean, self.cart_Twhiten, self.cart_Tblacken, self.std = pca(X0[:, self.cart_indices],
                                                                                keepdims=self.cart_indices.size-remove_dof)
        if np.any(self.std <= 0):
            raise ValueError('Cannot construct whiten layer because trying to keep nonpositive eigenvalues.')
        self.pca_log_det_xz = -np.sum(np.log(self.std))
        # Compute IC moments for normalization
        self.ic_means, self.ic_stds, self.torsion_cut = icmoments(self.Z_indices, X0=X0ic, torsion_cut=torsion_cut)


    @classmethod
    def from_dict(cls, D):
        ic_means = D['ic_means']
        ic_stds = D['ic_stds']
        torsion_cut = D['torsion_cut']
        cart_atom_indices = D['cart_atom_indices']
        cart_X0mean = D['cart_X0mean']
        cart_Twhiten = D['cart_Twhiten']
        cart_Tblacken = D['cart_Tblacken']
        Z_indices = D['Z_indices']
        dim = 3 * (cart_atom_indices.size + Z_indices.shape[0])
        c = cls(cart_atom_indices, Z_indices, X0=np.random.randn(2*dim, dim))
        c.cart_X0mean = tf.constant(cart_X0mean)
        c.cart_Twhiten = tf.constant(cart_Twhiten)
        c.cart_Tblacken = tf.constant(cart_Tblacken)
        c.ic_means = ic_means
        c.ic_stds = ic_stds
        c.torsion_cut = torsion_cut
        # optional
        if 'pca_log_det_xz' in D:
            pca_log_det_xz = D['pca_log_det_xz']
            c.pca_log_det_xz = pca_log_det_xz
        else:
            print('WARNING: Deprecated BG does not have a PCA log det Jacobian saved. Will set it to 0 and carry on ...')
            c.pca_log_det_xz = 0.0
        return c

    def to_dict(self):
        D = {}
        D['ic_means'] = self.ic_means
        D['ic_stds'] = self.ic_stds
        D['torsion_cut'] = self.torsion_cut
        D['cart_atom_indices'] = self.cart_atom_indices
        D['cart_X0mean'] = keras.backend.eval(self.cart_X0mean)
        D['cart_Twhiten'] = keras.backend.eval(self.cart_Twhiten)
        D['cart_Tblacken'] = keras.backend.eval(self.cart_Tblacken)
        D['pca_log_det_xz'] = self.pca_log_det_xz
        D['Z_indices'] = self.Z_indices
        return D

    def x2z(self, x):
        # split off Cartesian coordinates and perform whitening on them
        x_cart = tf.gather(x, self.cart_indices, axis=1)
        z_cart_signal = tf.matmul(x_cart - self.cart_X0mean, self.cart_Twhiten)

        # compute and normalize internal coordinates
        z_ics = xyz2ic_tf(x, self.Z_indices, torsion_cut=self.torsion_cut)
        z_ics_norm = (z_ics - self.ic_means) / self.ic_stds

        # concatenate the output
        z = tf.concat([z_cart_signal, z_ics_norm], axis=1)

        return z

    def z2x(self, z):
        # split off Cartesian block and unwhiten it
        dim_cart_signal = self.cart_indices.size-self.remove_dof
        z_cart_signal = z[:, :dim_cart_signal]
        x_cart = tf.matmul(z_cart_signal, self.cart_Tblacken) + self.cart_X0mean
        # split by atom
        #xyz = [x_cart[:, 3*i:3*(i+1)] for i in range(self.cart_atom_indices.size)]
        batchsize = tf.shape(z)[0]
        xyz = tf.reshape(x_cart, (batchsize, self.cart_atom_indices.size, 3))

        def _angle_loss(angle):
            positive_loss = tf.reduce_sum(tf.where(
                angle > 180, angle - 180, tf.zeros_like(angle)) ** 2, axis=-1)
            negative_loss = tf.reduce_sum(tf.where(
                angle < -180, angle + 180, tf.zeros_like(angle)) ** 2, axis=-1)
            return positive_loss + negative_loss

        # split off Z block
        z_ics_norm = z[:, dim_cart_signal:self.dim-self.remove_dof]
        z_ics = z_ics_norm * self.ic_stds + self.ic_means

        n_internal = self.dim - dim_cart_signal - self.remove_dof
        angle_idxs = np.arange(n_internal // 3) * 3 + 1
        torsion_idxs = np.arange(n_internal // 3) * 3 + 2

        angles = tf.gather(z_ics, angle_idxs, axis=-1)
        angle_loss = _angle_loss(angles)

        torsions = tf.gather(z_ics, torsion_idxs, axis=-1)
        torsions -= 180 + self.torsion_cut 
        angle_loss += _angle_loss(torsions)


        # reconstruct remaining atoms using ICs
        #ics2xyz_local(z_ics, self.Z_indices, self.index2order, xyz)
        xyz, _ = ics2xyz_local_log_det_jac_decomposed(z_ics, self.batchwise_Z_indices, xyz, self.index2order)

        # reorder and concatenate all atom coordinates
        x = tf.reshape(tf.gather(xyz, self.index2order, axis=1), (batchsize, -1))
        #xyz = [xyz[i] for i in self.index2order]
        #x = tf.concat(xyz, axis=1)

        return x, angle_loss

    def x2z_jacobian(self, x):
        # IC part
        log_det_jac = xyz2ic_log_det_jac(x, self.Z_indices, eps=self.jacobian_regularizer)
        # Add PCA part
        log_det_jac += self.pca_log_det_xz
        # reshape to (batchsize, 1)
        log_det_jac = tf.reshape(log_det_jac, (-1, 1))
        return log_det_jac

    def z2x_jacobian(self, z):
        # split off Cartesian block and unwhiten it
        dim_cart_signal = self.cart_indices.size-self.remove_dof
        z_cart_signal = z[:, :dim_cart_signal]
        x_cart = tf.matmul(z_cart_signal, self.cart_Tblacken) + self.cart_X0mean
        # split by atom
        #xyz = [x_cart[:, 3*i:3*(i+1)] for i in range(self.cart_atom_indices.size)]
        batchsize = tf.shape(z)[0]
        xyz = tf.reshape(x_cart, (batchsize, self.cart_atom_indices.size, 3))

        # split off Z block
        z_ics_norm = z[:, dim_cart_signal:self.dim-self.remove_dof]
        z_ics = z_ics_norm * self.ic_stds + self.ic_means
        #log_det_jac = ics2xyz_local_log_det_jac(z_ics, self.Z_indices, self.index2order, xyz)
        _, log_det_jac = ics2xyz_local_log_det_jac_decomposed(z_ics, self.batchwise_Z_indices, xyz, self.index2order, eps=self.jacobian_regularizer)
        # Add PCA part
        log_det_jac -= self.pca_log_det_xz
        # reshape to (batchsize, 1)
        log_det_jac = tf.reshape(log_det_jac, (-1, 1))
        return log_det_jac

