__author__ = "Vy Bui"
__email__ = "01bui@cua.edu"

import numpy as np
import scipy as sp
import scipy.interpolate
from scipy import ndimage as ndi
from skimage import segmentation
import SimpleITK as sitk

def spherical_refine_seg(spherical_gs_grid=None, spherical_target_edge=None,
                         low_threshold=None, high_threshold=None,
                         out_dir=None, DEBUG = 0, name=None):

    orig_spherical_grid = np.rot90(spherical_gs_grid, k=3, axes=(1, 2))
    spherical_target_edge = np.rot90(spherical_target_edge, k=3, axes=(1, 2))

    KERNEL_SIZE = 2
    spherical_gs_grid = ndi.filters.median_filter(orig_spherical_grid,
                                                  footprint=np.ones((KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE)))

    # Surface fitting
    fit_edge = np.zeros(orig_spherical_grid.shape, dtype=np.uint16)

    if low_threshold != None and high_threshold != None:
        threshold_mask = np.zeros(spherical_gs_grid.shape, dtype=np.uint8)  # *np.min(spherical_grid_shift)
        if high_threshold != None:
            idx = np.where((spherical_gs_grid > low_threshold) & (spherical_gs_grid < high_threshold))
        else:
            idx = np.where(spherical_gs_grid > low_threshold)
        threshold_mask[idx] = 1

        KERNEL_SIZE = 2
        med_mask = ndi.filters.median_filter(threshold_mask, footprint=np.ones((KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE)))

        radius = 2
        r2 = np.arange(-radius, radius + 1) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        s_e_sphere = (dist2 <= radius ** 2).astype(np.int)
        morph_mask = ndi.binary_erosion(med_mask, s_e_sphere).astype(dtype=np.uint8)

        label_objects, nb_labels = ndi.label(morph_mask)
        sizes = np.bincount(label_objects.ravel())
        sorted_sizes = np.sort(sizes)[::-1]
        if len(sizes) > 2:
            mask_sizes = (sizes == np.max(sorted_sizes[1:-1]))
        elif len(sizes) == 2:
            mask_sizes = (sizes == sorted_sizes[1])
        elif len(sizes) == 1:
            mask_sizes = (sizes == sorted_sizes[0])
        morph_mask = mask_sizes[label_objects].astype(dtype=np.uint8)

        radius = 2
        r2 = np.arange(-radius, radius + 1) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        s_e_sphere = (dist2 <= radius ** 2).astype(np.int)
        morph_mask = ndi.binary_dilation(morph_mask, s_e_sphere).astype(dtype=np.uint8)

        edge = segmentation.find_boundaries(morph_mask, mode='outer').astype(np.uint16)

        # Make the lv edge thinner
        # Keep furthest point along r axis
        thin_edge = np.zeros(edge.shape, dtype=np.uint8)
        for img_idx in range(edge.shape[0]):
            xnew = []
            im = edge[img_idx, :, :]
            x, y = np.where(im > 0)
            U = np.unique(y)
            for i in U:
                xnew = np.max(x[np.where(y == i)])
                ynew = np.unique(y[np.where(y == i)])
                thin_edge[img_idx, xnew, ynew] = 1

        # Keep furthest point along r axis
        extreme_edge = np.zeros(thin_edge.shape, dtype=np.uint8)
        for img_idx in range(thin_edge.shape[0]):
            xnew = []
            im = thin_edge[img_idx, :, :]
            x, y = np.where(im > 0)
            U = np.unique(y)
            for i in U:
                xnew = np.max(x[np.where(y == i)])
                ynew = np.unique(y[np.where(y == i)])
                extreme_edge[img_idx, xnew, ynew] = 1

        if DEBUG == 1:
            general_itk = sitk.GetImageFromArray(extreme_edge.astype(dtype=np.int16))
            sitk.WriteImage(general_itk, out_dir+'extreme_edge_'+name+'.tiff')

        z, x, y = np.where(extreme_edge > 0)
        ny = y[0::20]
        nx = x[0::20]
        nz = z[0::20]
        fit_edge[nz, nx, ny] = 1

    z, x, y = np.where(spherical_target_edge > 0)
    ny = y[0::10]
    nx = x[0::10]
    nz = z[0::10]
    fit_edge[nz, nx, ny] = 1

    nz, nx, ny = np.where(fit_edge > 0)

    data = np.c_[ny, nz, nx]
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    YI, ZI = np.meshgrid(np.linspace(mn[0], mx[0], 200), np.linspace(mn[1], mx[1], 200))

    spline = sp.interpolate.Rbf(nz, ny, nx, function='linear', smooth=4)
    # 'multiquadric', 'inverse', 'gaussian' 'linear', 'cubic', 'quintic', 'thin_plate'
    XI = spline(ZI, YI)

    XI[XI >= fit_edge.shape[1]] = fit_edge.shape[1] - 1
    XI[XI < 0] = 0

    XI = XI.astype(dtype=np.uint16)
    YI = YI.astype(dtype=np.uint16)
    ZI = ZI.astype(dtype=np.uint16)
    ZI, XI, YI = ZI.flatten(), XI.flatten(), YI.flatten()

    fit_mask_bspl = np.zeros(fit_edge.shape, dtype=np.int16)
    temp = np.zeros(fit_edge.shape, dtype=np.int16)
    temp[nz, nx, ny] = 1
    fit_mask_bspl[ZI, XI, YI] = 1

    if DEBUG == 1:
        ### Write tiff color mask
        img_rgb = np.zeros(shape=(fit_edge.shape[0], fit_edge.shape[1], fit_edge.shape[2], 3), dtype=np.uint8)
        """z, x, y = np.where(lv_edge == 1)
        img_rgb[z, x, y, 0] = 128
        img_rgb[z, x, y, 1] = 128
        img_rgb[z, x, y, 2] = 128"""
        """z, x, y = np.where(fit_mask == 1)  #Red
        img_rgb[z, x, y, 0] = 255
        img_rgb[z, x, y, 1] = 0
        img_rgb[z, x, y, 2] = 0"""
        z, x, y = np.where(temp == 1)  # Yellow
        img_rgb[z, x, y, 0] = 255
        img_rgb[z, x, y, 1] = 255
        img_rgb[z, x, y, 2] = 0
        z, x, y = np.where(fit_mask_bspl == 1)  # Blue
        img_rgb[z, x, y, 0] = 0
        img_rgb[z, x, y, 1] = 0
        img_rgb[z, x, y, 2] = 255
        img_rgb = img_rgb.astype(dtype=np.uint8)

        general_itk = sitk.GetImageFromArray(fit_mask_bspl)
        sitk.WriteImage(general_itk, out_dir +'gray_'+name+'.tif')

        general_itk = sitk.GetImageFromArray(img_rgb)
        sitk.WriteImage(general_itk, out_dir +'fit_mask_'+name+'.tiff')

    z, x, y = np.where(fit_mask_bspl > 0)
    B = np.copy(orig_spherical_grid)
    B[z, x, y] = np.min(B)

    if DEBUG == 1:
        general_itk = sitk.GetImageFromArray(B.astype(dtype=np.int16))
        sitk.WriteImage(general_itk, out_dir +'bspline_in_sph_'+name+'.tiff')
        sitk.WriteImage(general_itk, out_dir +'bspline_in_sph_'+name+'.nii.gz')

    z, x, y = np.where(fit_edge > 0)
    orig_spherical_grid[z, x, y] = np.min(orig_spherical_grid)

    # Rotate back to original orientation
    orig_spherical_grid = np.rot90(orig_spherical_grid, k=1, axes=(1, 2))
    fit_edge = np.rot90(fit_edge, k=1, axes=(1, 2))
    fit_mask_bspl = np.rot90(fit_mask_bspl, k=1, axes=(1, 2))

    return orig_spherical_grid, fit_edge, fit_mask_bspl