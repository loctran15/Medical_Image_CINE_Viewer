
import numpy as np
from Dicom_helper import dcm_reader
import os
from src_CMAS.cmacs_regis import deeds_regis
import SimpleITK as sitk
from src_CMAS.cmacs_bbox_detection import bbox_detection
from src_CMAS.cmacs_mcs import get_bounding_vols



def get_checkerboard(m_vol, f_vol, warp):
    ezdim, exdim, eydim = m_vol.shape
    # vol1h = hist_equal(vol1, percent=1)
    vol1h = f_vol #exposure.equalize_hist(f_vol, nbins=np.unique(f_vol))
    # volRh = hist_equal(volR, percent=1)
    volRh = warp + 255 #exposure.equalize_hist(warp, nbins=np.unique(warp))
    # vol2h = hist_equal(vol2, percent=1)
    vol2h = m_vol + 255 #exposure.equalize_hist(m_vol, nbins=np.unique(m_vol))

    # volCB1 = intarr(exdim,eydim,ezdim)
    volCB1 = np.zeros((ezdim,exdim,eydim))
    # volCB2 = intarr(exdim,eydim,ezdim)
    volCB2 = np.zeros((ezdim,exdim,eydim))
    # volCB3 = intarr(exdim,eydim,ezdim)
    volCB3 = np.zeros((ezdim,exdim,eydim))
    # bsz = 9
    bsz = 9
    # xsz = fix(exdim/bsz)
    xsz = int(exdim/bsz)
    # ysz = fix(eydim/bsz)
    ysz = int(eydim/bsz)
    # xsf = (exdim - (xsz) * bsz) / 2
    xsf = int((exdim - (xsz) * bsz) / 2)
    # ysf = (eydim - (ysz) * bsz) / 2
    ysf = int((eydim - (ysz) * bsz) / 2)
    # for x=0,bsz-1 do $
    for x in range(0, bsz):
    # for y=0,bsz-1 do $
      for y in range(0,bsz):
    #   volCB1[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *] = ((y + x*bsz) mod 2) ? $
    #    volRh[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *] : $
    #    vol1h[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *]
        if(((y + x*bsz) % 2) == 0):
          volCB1[:, ysf + y * ysz:min([ysf + (y + 1) * ysz, eydim - 1]), xsf + x * xsz:min([xsf + (x + 1) * xsz, exdim - 1])] = volRh[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])]
        else:
          volCB1[:,ysf + y * ysz:min([ysf + (y + 1) * ysz, eydim - 1]), xsf + x * xsz:min([xsf + (x + 1) * xsz, exdim - 1])] = vol1h[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])]

    # for x=0,bsz-1 do $
    for x in range(0, bsz):
    # for y=0,bsz-1 do $
      for y in range(0, bsz):
    #   volCB2[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *] = ((y + x*bsz) mod 2) ? $
    #    vol1h[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *] : $
    #    volRh[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *]
          if (((y + x*bsz) % 2) == 0):
            volCB2[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])] = vol1h[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])]
          else:
            volCB2[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])] = volRh[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])]

    #
    # for x=0,bsz-1 do $
    for x in range(0, bsz):
    # for y=0,bsz-1 do $
      for y in range(0, bsz):
    #   volCB3[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *] = ((y + x*bsz) mod 2) ? $
    #    vol1h[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *] : $
    #    vol2h[xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1]), ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]), *]
        if (((y + x*bsz) % 2) == 0):
          volCB3[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])] = vol1h[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])]
        else:
          volCB3[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])] = vol2h[:, ysf+y*ysz:min([ysf+(y+1)*ysz,eydim-1]),xsf+x*xsz:min([xsf+(x+1)*xsz,exdim-1])]

    #
    # volDBG = intarr(3*exdim+6,2*eydim+6,ezdim) + 255
    volDBG = np.zeros((ezdim,2*eydim+6,3*exdim+6)) + 255
    # volDBG[        2:  exdim+1,       2:  eydim+1, *] = vol1h
    volDBG[:,       2:  eydim+2,        2:  exdim+2] = vol1h
    # volDBG[  exdim+4:2*exdim+3,       2:  eydim+1, *] = volRh
    volDBG[:,       2:  eydim+2,  exdim+4:2*exdim+4] = volRh
    # volDBG[2*exdim+6:3*exdim+5,       2:  eydim+1, *] = vol2h
    volDBG[:,       2:  eydim+2, 2*exdim+6:3*exdim+6] = vol2h
    # volDBG[        2:  exdim+1, eydim+4:2*eydim+3, *] = volCB1
    volDBG[:, eydim+4:2*eydim+4,        2:  exdim+2] = volCB1   #warp and fixed
    # volDBG[  exdim+4:2*exdim+3, eydim+4:2*eydim+3, *] = volCB2
    volDBG[:, eydim+4:2*eydim+4,  exdim+4:2*exdim+4] = volCB2
    # volDBG[2*exdim+6:3*exdim+5, eydim+4:2*eydim+3, *] = volCB3
    volDBG[:, eydim+4:2*eydim+4, 2*exdim+6:3*exdim+6] = volCB3 #fixed and moving

    volDBG = volDBG.astype(np.int16)
    return volDBG


if __name__ == "__main__":
    #testing purpose
    root_dir = "D:/TRANL/DATA/SQUEEZ/processed_data/SQUEEZ-With-Labels/SQUEEZ-0001-1"

    m_vol_index = 2
    f_vol_index = 3

    moving_path = os.path.join(root_dir, f"1.2.392.200036.9116.2.2462354099.1572330400.1.1352500001.{m_vol_index}")
    fixed_path =  os.path.join(root_dir, f"1.2.392.200036.9116.2.2462354099.1572330400.1.1352500001.{f_vol_index}")

    m_vol = np.copy(dcm_reader.read_cases(moving_path)).astype(np.int16)

    f_vol = np.copy(dcm_reader.read_cases(fixed_path)).astype(np.int16)

    #bbox, _, _ = bbox_detection(volume_dir=f_vol, check_dir=3, debug_path=None, DEBUG=0)

    #f_padvol, m_padvol = get_bounding_vols(f_vol,bbox,m_vol,bbox)

    _,warp = deeds_regis(f_vol, m_vol, path = "../CT_SEG/CMACS_EXE", level = 4, alpha_ = 2.5)

    #m_vol = np.full(m_vol.shape,fill_value=-2048,dtype=np.int16)
    #f_vol = np.full(f_vol.shape,fill_value=-2048,dtype=np.int16)
    #warp  = np.full(m_vol.shape,fill_value=-2048,dtype=np.int16)

    #z, x, y = np.where(bbox > 0)
    #m_vol[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = m_padvol
    #f_vol[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = f_padvol
    #warp[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = warp_pad

    cb = get_checkerboard(m_vol,f_vol,warp)
    vol_1mm_itk = sitk.GetImageFromArray(cb)
    vol_1mm_itk.SetSpacing([1, 1, 1])
    sitk.WriteImage(vol_1mm_itk, f"D:/TRANL/DATA/deed_evaluation/m_{m_vol_index}_f_{f_vol_index}_deed_L4_a2.5.nii.gz")