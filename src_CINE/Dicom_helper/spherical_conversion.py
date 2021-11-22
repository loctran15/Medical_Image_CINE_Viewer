__author__ = "Vy Bui"
__email__ = "01bui@cua.edu"

import numpy as np
from scipy.ndimage import map_coordinates

def index_coords_cart(data, origin=None):
    nx, ny, nz = data.shape[:3]
    if origin is None:
        origin_x, origin_y, origin_z = nx // 2, ny // 2, nz // 2
    else:
        origin_x, origin_y, origin_z = origin

    x, y, z = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)), np.arange(float(nz)))

    x -= origin_x
    y -= origin_y
    z -= origin_z
    return x, y, z

def cart2spherical(x, y, z):
    hxy = np.hypot(x, y)
    # r = np.hypot(hxy, z)
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arctan2(hxy, z)
    return r, theta, phi

def spherical2cart(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def binary_reproject_cart_into_sph(data, origin=None):

    x, y, z = index_coords_cart(data, origin=origin)  # (x,y,z) coordinates of each pixel

    r, theta, phi = cart2spherical(x, y, z)  # convert to spherical coordinates

    # step sampling (pick to fixed size)
    dr = (r.max() - r.min())/data.shape[0]
    dt = 2.*np.pi/data.shape[1]
    dph= 1.*np.pi/data.shape[2]

    """dr = 1.
    dt=np.pi/180
    dph=np.pi/360"""

    # finding grid length
    """nr = np.int(np.ceil((r.max() - r.min()) / dr))
    nt = np.int(np.ceil((theta.max() - theta.min()) / dt))
    nph = np.int(np.ceil((phi.max() - phi.min()) / dph))"""
    nr, nt, nph = data.shape[:3]

    # Make a regular (in polar space) grid based on the min and max r, theta, phi
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    phi_i = np.linspace(phi.min(), phi.max(), nph, endpoint=False)

    # meskrid with spherical coordinates
    r_grid, theta_grid, phi_grid = np.meshgrid(r_i, theta_i, phi_i)
    X, Y, Z = spherical2cart(r_grid, theta_grid, phi_grid)

    if origin is None:
        X += nr // 2
        Y += nt // 2
        Z += nph // 2
    else:
        X += origin[0]  # We need to shift the origin
        Y += origin[1]
        Z += origin[2]  # back to the up-left corner...

    xi, yi, zi = X.flatten(), Y.flatten(), Z.flatten()
    coords = np.vstack((xi, yi, zi))  # (map_coordinates requires a 3xn array)
    K = map_coordinates(data, coords, order=0, mode='constant', cval=np.min(data))
    #K = map_coordinates(data, coords, order=3, mode='nearest')

    output = K.reshape((nr, nt, nph))  # Follow Z, X, Y order?
    return output

def reproject_cart_into_sph(data, origin=None):
    x, y, z = index_coords_cart(data, origin=origin)  # (x,y,z) coordinates of each pixel

    r, theta, phi = cart2spherical(x, y, z)  # convert to spherical coordinates

    # step sampling (pick to fixed size)
    dr = (r.max() - r.min())/data.shape[0]
    dt = 2.*np.pi/data.shape[1]
    dph= 1.*np.pi/data.shape[2]

    """dr = 1.
    dt=np.pi/180
    dph=np.pi/360"""

    # finding grid length
    """nr = np.int(np.ceil((r.max() - r.min()) / dr))
    nt = np.int(np.ceil((theta.max() - theta.min()) / dt))
    nph = np.int(np.ceil((phi.max() - phi.min()) / dph))"""
    nr, nt, nph = data.shape[:3]

    # Make a regular (in polar space) grid based on the min and max r, theta, phi
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    phi_i = np.linspace(phi.min(), phi.max(), nph, endpoint=False)

    # meskrid with spherical coordinates
    r_grid, theta_grid, phi_grid = np.meshgrid(r_i, theta_i, phi_i)
    X, Y, Z = spherical2cart(r_grid, theta_grid, phi_grid)

    if origin is None:
        X += nr // 2
        Y += nt // 2
        Z += nph // 2
    else:
        X += origin[0]  # We need to shift the origin
        Y += origin[1]
        Z += origin[2]  # back to the up-left corner...

    xi, yi, zi = X.flatten(), Y.flatten(), Z.flatten()
    coords = np.vstack((xi, yi, zi))  # (map_coordinates requires a 3xn array)
    K = map_coordinates(data, coords, order=3, mode='nearest')

    output = K.reshape((nr, nt, nph))  # Follow Z, X, Y order?
    return output

def reproject_sph_into_cart(data, origin=None, orig_data=None):
    x, y, z = index_coords_cart(data, origin=origin)  # (x,y,z) coordinates of each pixel

    r, theta, phi = cart2spherical(x, y, z)  # convert to spherical coordinates

    nr, nt, nph = orig_data.shape[:3]  # size is fixed

    # get correspoding pixel coordicates in r, t, ph volume
    X = (r - r.min()) * nr / (r.max() - r.min())
    Y = (theta - theta.min()) * nt / (theta.max() - theta.min())
    Z = (phi - phi.min()) * nph / (phi.max() - phi.min())

    xi, yi, zi = X.flatten(), Y.flatten(), Z.flatten()
    coords = np.vstack((yi, xi, zi))  # (map_coordinates requires a 3xn array)

    K = map_coordinates(data, coords, order=3, mode='nearest')
    output = K.reshape((nr, nt, nph))

    return output