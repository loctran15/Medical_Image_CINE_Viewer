__author__ = "Vy Bui"
__email__ = "01bui@cua.edu"

from scipy.spatial import ConvexHull, Delaunay
from skimage import exposure
from .spherical_processing import spherical_refine_seg
from .spherical_conversion import reproject_cart_into_sph, reproject_sph_into_cart, binary_reproject_cart_into_sph
from scipy import ndimage as ndi
from skimage import segmentation
import numpy as np

# Label Dictionary
label_dict = {
    "BOX"  : 1,
    "WH"   : 2,
    "LUNG" : 3,
    "LVM"  : 5,
    "LV"   : 6,
    "AO"   : 9,
    "LIVER": 10,
    "DAS"  : 11,
    "RV"   : 12,
    "CW"   : 13,
    "PV"   : 14,
    "LA"   : 15,
    "LAA"  : 16,
    "RA"   : 19,
    "IVC"  : 20,
    "PA"   : 21,
    "SVC"  : 22,
    "SPINE": 23,
}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def maxip(img, slices_num = 15):
    img_shape = img.shape
    mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        mip[i,:,:] = np.amax(img[start:i + 1], 0)
    return mip

def minip(img, slices_num = 15):
    img_shape = img.shape
    mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        mip[i,:,:] = np.amin(img[start:i + 1], 0)
    return mip

# Estimates gaussian distribution parameters from data using ML
def gaussian_estimation(vector):
    mu = np.mean(vector)
    sig = np.std(vector)

    return (mu,sig)

# Adjusts the data so it forms a gaussian with mean of 0 and std of 1
def gaussian_normalization(vector, char = None):

    if char is None:
        mu , sig = gaussian_estimation(vector)
    else:
        mu = char[2]
        sig = char[3]

    normalized = (vector-mu)/(sig)

    return normalized

# https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
def CDF(x, max_i = 100):
    sum = x
    value = x
    for i in np.arange(max_i)+1:
        value = value*x*x/(2.0*i+1)
        sum = sum + value

    return 0.5 + (sum/np.sqrt(2*np.pi))*np.exp(-1*(x*x)/2)

def gaussian_to_uniform(vector, if_normal = False):

    if (if_normal == False):
        vector = gaussian_normalization(vector)

    uni = np.apply_along_axis(CDF, 0, vector)

    return uni

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)

def match_histograms(image, reference, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape
    image_dtype = image.dtype

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched

def reject_outliers(data=None, threshold=None, m=2):
    temp = data[data > np.min(data)]
    good_values = temp[abs(temp - np.median(temp)) < m * np.std(temp)]
    good_values = good_values[good_values >= threshold]
    idx = np.isin(data, good_values)
    return good_values

def remove_points(data=None, mask=None, threshold=None, m=2):
    gval = np.ones(mask.shape, dtype=np.int16)*np.min(data)
    z, x, y = np.where(mask > 0)
    gval[z, x, y] = data[z, x, y]
    good_values = reject_outliers(gval, threshold=threshold, m=m)
    final_mask = np.zeros(mask.shape, dtype=np.uint16)
    idx = np.where((gval >= np.min(good_values)) & (gval <= np.max(good_values)))
    final_mask[idx] = 1
    return final_mask

def remove_points_over_percentile(data=None, mask=None, bool_mask = 0, low=5, high=95):
    gval = np.ones(data.shape, dtype=np.int16)*np.min(data)
    if bool_mask == 1:
        z, x, y = np.where(mask > 0)
        gval[z, x, y] = data[z, x, y]
    else:
        gval = data
    low_value = np.percentile(gval[gval > np.min(gval)], low)
    high_value = np.percentile(gval[gval > np.min(gval)], high)
    final_mask = np.zeros(data.shape, dtype=np.uint16)
    idx = np.where((gval >= low_value) & (gval <= high_value))
    final_mask[idx] = 1
    return final_mask

def flood_fill_hull(volume):
    points = np.transpose(np.where(volume))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(volume.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(volume.shape)
    out_img[out_idx] = 1
    return out_img.astype(dtype=np.uint16)

def spherical_smoothing_mask(orig_volume, target_mask, low_threshold=None, high_threshold=None, out_dir=None, DEBUG = None, name=None):
    unnorm = target_mask * orig_volume
    index = np.where(target_mask < np.max(target_mask))
    unnorm[index] = np.min(orig_volume)
    mask_gvol = unnorm.astype(np.int16)

    ZZ = mask_gvol.shape[0]
    XX = mask_gvol.shape[1]
    YY = mask_gvol.shape[2]

    cmass = ndi.measurements.center_of_mass(target_mask > 0)

    com = []
    com.append(cmass[1])
    com.append(cmass[2])
    com.append(cmass[0])

    ##############################################
    # Cartesian to Spherical Mapping
    ##############################################
    spherical_gs_grid_in = np.empty(shape=(XX, YY, ZZ), dtype=np.int16)
    z, x, y = np.where(mask_gvol >= np.min(mask_gvol))
    spherical_gs_grid_in[x, y, z] = mask_gvol[z, x, y]
    spherical_gs_grid = reproject_cart_into_sph(spherical_gs_grid_in, origin=com)
    spherical_gs_grid_out = np.empty(shape=(ZZ, XX, YY), dtype=np.int16)
    x, y, z = np.where(spherical_gs_grid >= np.min(spherical_gs_grid))
    spherical_gs_grid_out[z, x, y] = spherical_gs_grid[x, y, z]

    ##############################################
    # Convert target edge to spherical space
    ##############################################
    target_edge = segmentation.find_boundaries(target_mask.astype(np.uint8), mode='inner').astype(np.uint16)
    spherical_target_edge_in = np.zeros(shape=(XX, YY, ZZ), dtype=np.uint16)
    z, x, y = np.where(target_edge > np.min(target_edge))
    spherical_target_edge_in[x, y, z] = target_edge[z, x, y]
    spherical_target_edge = binary_reproject_cart_into_sph(spherical_target_edge_in, origin=com)
    spherical_target_edge_out = np.zeros(shape=(ZZ, XX, YY), dtype=np.uint16)
    x, y, z = np.where(spherical_target_edge > np.min(spherical_target_edge))
    spherical_target_edge_out[z, x, y] = spherical_target_edge[x, y, z]

    ##############################################
    # Refine segmentation in Spherical space
    ##############################################
    _, _, fit_mask_bspl = spherical_refine_seg(spherical_gs_grid=spherical_gs_grid_out,
                                               spherical_target_edge=spherical_target_edge_out,
                                               low_threshold=low_threshold, high_threshold=high_threshold,
                                               out_dir=out_dir, DEBUG=DEBUG, name=name)

    bspl_mask_in = np.empty(shape=(XX, YY, ZZ), dtype=np.int16)
    mask_bspl = np.zeros(fit_mask_bspl.shape, dtype=np.int16)
    for img_idx in range(fit_mask_bspl.shape[0]):
        im = fit_mask_bspl[img_idx, :, :]
        x, y = np.where(im > 0)
        for i in range(len(x)):
            xnew = []
            ynew = []
            for j in range(0, y[i], 1):
                xnew.append(x[i])
                ynew.append(j)
                mask_bspl[img_idx, xnew, ynew] = 1
    z, x, y = np.where(mask_bspl >= 0)
    bspl_mask_in[x, y, z] = mask_bspl[z, x, y]

    ##############################################
    # Spherical to Cartesian Mapping
    ##############################################
    refine_bspl_out = reproject_sph_into_cart(bspl_mask_in, origin=com, orig_data=spherical_gs_grid_in)
    refine_bspl = np.empty(shape=(ZZ, XX, YY), dtype=np.int16)
    y, x, z = np.where(refine_bspl_out >= np.min(refine_bspl_out))
    refine_bspl[z, x, y] = refine_bspl_out[y, x, z]
    return refine_bspl

def histogram_equalization(volume, mask=None, nbins=256):
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        cdf, bin_centers = exposure.cumulative_distribution(volume[mask], nbins)
    else:
        cdf, bin_centers = exposure.cumulative_distribution(volume, nbins)
    out = np.interp(volume.flat, bin_centers, cdf)
    return out

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))