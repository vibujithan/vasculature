import numpy as np
from scipy import interpolate
from scipy.ndimage import filters
from scipy.stats import circmean


def structure_tensor(volume, sigma, rho):
    # Make sure it's a Numpy array.
    volume = np.asarray(volume)

    # Computing derivatives (scipy implementation truncates filter at 4 sigma).
    Vx = filters.gaussian_filter(volume, sigma, order=[0, 0, 1], mode='nearest')
    Vy = filters.gaussian_filter(volume, sigma, order=[0, 1, 0], mode='nearest')
    Vz = filters.gaussian_filter(volume, sigma, order=[1, 0, 0], mode='nearest')

    S = np.empty(volume.shape + (3, 3), dtype=volume.dtype)

    # Integrating elements of structure tensor (scipy uses sequence of 1D).
    grad = np.zeros(volume.shape, dtype=volume.dtype)
    tmp = np.empty(volume.shape, dtype=volume.dtype)

    np.multiply(Vx, Vx, out=tmp)
    filters.gaussian_filter(tmp, rho, mode='nearest', output=S[..., 0, 0])
    grad += tmp

    np.multiply(Vy, Vy, out=tmp)
    filters.gaussian_filter(tmp, rho, mode='nearest', output=S[..., 1, 1])
    grad += tmp

    np.multiply(Vz, Vz, out=tmp)
    filters.gaussian_filter(tmp, rho, mode='nearest', output=S[..., 2, 2])
    grad += tmp

    np.multiply(Vx, Vy, out=tmp)
    filters.gaussian_filter(tmp, rho, mode='nearest', output=S[..., 0, 1])

    np.multiply(Vx, Vz, out=tmp)
    filters.gaussian_filter(tmp, rho, mode='nearest', output=S[..., 0, 2])

    np.multiply(Vy, Vz, out=tmp)
    filters.gaussian_filter(tmp, rho, mode='nearest', output=S[..., 1, 2])

    return S, grad


def circ_average(vectors):
    Vz = vectors[..., 2]
    Vy = vectors[..., 1]
    Vx = vectors[..., 0]

    beta = np.arccos(Vz)
    gamma = np.arctan2(Vy, Vx)

    # beta = np.mod(beta, np.pi)
    # gamma = np.mod(gamma, np.pi)

    beta_mean = circmean(beta)
    gamma_mean = circmean(gamma)

    z = np.cos(beta_mean)
    x = np.sin(beta_mean) * np.cos(gamma_mean)
    y = np.sin(beta_mean) * np.sin(gamma_mean)

    return x, y, z


def positive_dominant_vector(vectors):
    max_id = np.argmax(abs(vectors), axis=1)
    ind = np.arange(0, len(vectors))
    return np.expand_dims(np.sign(vectors[ind, max_id]), axis=1) * vectors


def rotate_vectors(vectors):
    Vz = vectors[..., 2]
    Vy = vectors[..., 1]
    Vx = vectors[..., 0]

    for i in range(0, len(Vz)):
        if Vz[i] > 0 or (Vz[i] == 0 and Vy[i] > 0):
            pass
        elif Vz[i] == 0 and Vy[i] < 0:
            Vx[i] = -Vx[i]
            Vy[i] = -Vy[i]
        elif Vz[i] == 0 and Vy[i] == 0:
            Vx[i] = np.abs(Vx[i])
        else:
            Vx[i] = -Vx[i]
            Vy[i] = -Vy[i]
            Vz[i] = -Vz[i]

    vectors[..., 2] = Vz
    vectors[..., 1] = Vy
    vectors[..., 0] = Vx

    return vectors


def mean_axis(vectors):
    no_of_vectors = vectors.shape[0]

    if no_of_vectors == 0:
        return [np.nan, np.nan, np.nan]

    vectors = np.transpose(vectors)
    pos_id = (vectors != 0).argmax(axis=0)
    ind = np.arange(0, no_of_vectors)
    pos_id_sign = np.sign(vectors[pos_id, ind])

    vectors = np.expand_dims(pos_id_sign, axis=0) * vectors

    sm = np.matmul(vectors, np.transpose(vectors)) / no_of_vectors
    eigen_values, eigen_vectors = np.linalg.eig(sm)

    idx = eigen_values.argsort()
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    if eigen_values[-1] > 0.1:
        print(eigen_vectors[:, -1])
        return eigen_vectors[:, -1]
    else:
        return [np.nan, np.nan, np.nan]


def make_volume(volume, block=20, interp=1):
    if interp != 1:
        myo = np.nan_to_num(volume)

        x = np.arange(0, myo.shape[2], 1)
        y = np.arange(0, myo.shape[1], 1)

        vec_0 = myo[0, :, :]
        vec_1 = myo[1, :, :]
        vec_2 = myo[2, :, :]

        f0 = interpolate.interp2d(x, y, vec_0, kind='cubic')
        f1 = interpolate.interp2d(x, y, vec_1, kind='cubic')
        f2 = interpolate.interp2d(x, y, vec_2, kind='cubic')

        xx = np.arange(0, myo.shape[2], interp)
        yy = np.arange(0, myo.shape[1], interp)

        new_0 = np.expand_dims(f0(xx, yy), axis=0)
        new_1 = np.expand_dims(f1(xx, yy), axis=0)
        new_2 = np.expand_dims(f2(xx, yy), axis=0)

        volume = np.concatenate((new_0, new_1, new_2), axis=0)
    else:
        volume = np.nan_to_num(volume)

    vol = np.zeros((block, block * volume.shape[1], block * volume.shape[2]), dtype='|b')
    t = np.arange(-8, 8)

    for i in range(0, volume.shape[1]):
        for j in range(0, volume.shape[2]):
            zp = block / 2 + t * np.nan_to_num(volume[2, i, j])
            xp = (block / 2 + t * np.nan_to_num(volume[1, i, j]))
            yp = block / 2 + t * np.nan_to_num(volume[0, i, j])
            vol[zp.astype(int), (i * block) + xp.astype(int), (j * block) + yp.astype(int)] = True
    return vol
