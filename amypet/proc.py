'''
Processing of PET images for AmyPET 
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging as log
import os
from pathlib import Path, PurePath

import numpy as np
from niftypet import nimpa

log.basicConfig(level=log.WARNING, format=nimpa.LOG_FORMAT)


# ========================================================================================
def extract_vois(impet, atlas, voi_dct, atlas_mask=None, outpath=None, output_masks=False):
    '''
    Extract VOI mean values from PET image `impet` using image labels `atals`.
    Both can be dictionaries, file paths or Numpy arrays.
    They have to be aligned and have the same dimensions.
    If path (output) is given, the ROI masks will be saved to file(s).
    Arguments:
        - impet:    PET image as Numpy array
        - atlas:  image of labels (integer values); the labels can come
                    from T1w-based parcellation or an atlas.
        - voi_dct:  dictionary of VOIs, with entries of labels creating
                    composite volumes
        - atlas_mask: masks the atlas with an additional maks, e.g., with the
                    grey matter probability mask.
        - output_masks: if `True`, output Numpy VOI masks in the output
                    dictionary
        - outpath:  if given as a folder path, the VOI masks will be saved
    '''

    # > assume none of the below are given
    # > used only for saving ROI mask to file if requested
    affine, flip, trnsp = None, None, None

    # ----------------------------------------------
    # PET
    if isinstance(impet, dict):
        im = impet['im']
        if 'affine' in impet:
            affine = impet['affine']
        if 'flip' in impet:
            flip = impet['flip']
        if 'transpose' in impet:
            trnsp = impet['transpose']

    elif isinstance(impet, (str, PurePath)) and os.path.isfile(impet):
        imd = nimpa.getnii(impet, output='all')
        im = imd['im']
        flip = imd['flip']
        trnsp = imd['transpose']

    elif isinstance(impet, np.ndarray):
        im = impet
    # ----------------------------------------------

    # ----------------------------------------------
    # LABELS
    if isinstance(atlas, dict):
        lbls = atlas['im']
        if 'affine' in atlas and affine is None:
            affine = atlas['affine']
        if 'flip' in atlas and flip is None:
            flip = atlas['flip']
        if 'transpose' in atlas and trnsp is None:
            trnsp = atlas['transpose']

    elif isinstance(atlas, (str, PurePath)) and os.path.isfile(atlas):
        prd = nimpa.getnii(atlas, output='all')
        lbls = prd['im']
        if affine is None:
            affine = prd['affine']
        if flip is None:
            flip = prd['flip']
        if trnsp is None:
            trnsp = prd['transpose']

    elif isinstance(atlas, np.ndarray):
        lbls = atlas

    # > get rid of NaNs if any in the parcellation/label image
    lbls[np.isnan(lbls)] = 0

    # > atlas mask
    if atlas_mask is not None:
        if isinstance(atlas_mask, (str, PurePath)) and os.path.isfile(atlas_mask):
            amsk = nimpa.getnii(atlas_mask)
        elif isinstance(atlas_mask, np.ndarray):
            amsk = atlas_mask
        else:
            raise ValueError('Incorrectly provided atlas mask')
    else:
        amsk = 1
    # ----------------------------------------------

    # ----------------------------------------------
    # > output dictionary
    out = {}

    log.debug('Extracting volumes of interest (VOIs):')
    for k, voi in enumerate(voi_dct):

        log.info(f'  VOI: {voi}')

        # > ROI mask
        rmsk = np.zeros(lbls.shape, dtype=bool)

        for ri in voi_dct[voi]:
            log.debug(f'   label{ri}')
            rmsk += np.equal(lbls, ri)

        # > apply the mask on mask
        if not isinstance(amsk, np.ndarray) and amsk==1:
            msk2 = rmsk
        else:
            msk2 = rmsk*amsk

        if outpath is not None and not isinstance(atlas, np.ndarray):
            nimpa.create_dir(outpath)
            fvoi = Path(outpath) / (str(voi) + '_mask.nii.gz')
            nimpa.array2nii(msk2, affine, fvoi,
                            trnsp=(trnsp.index(0), trnsp.index(1), trnsp.index(2)), flip=flip)
        else:
            fvoi = None
        
        vxsum = np.sum(msk2)

        if im.ndim==4:
            nfrm = im.shape[0]
            emsum = np.zeros(nfrm, dtype=np.float64)
            for fi in range(nfrm):
                emsum[fi] = np.sum(im[fi,...].astype(np.float64) * msk2)
        
        elif im.ndims==3:
            emsum = np.sum(im.astype(np.float64)*msk2)
        
        else:
            raise ValueError('unrecognised image shape or dimensions')

        
        out[voi] = {'vox_no': vxsum, 'sum': emsum, 'avg': emsum / vxsum, 'fvoi': fvoi}

        if output_masks:
            out[voi]['roimsk'] = msk2

    # ----------------------------------------------

    return out