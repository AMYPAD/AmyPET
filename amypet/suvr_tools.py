import os
import sys
import re

from pathlib import Path, PurePath

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from subprocess import run
import dcm2niix

#from miutil.fdio import hasext
from niftypet import nimpa

nifti_ext = ('.nii', '.nii.gz')
dicom_ext = ('.DCM', '.dcm', '.img', '.IMG')

# ========================================================================================
def extract_vois(impet, imlabel, voi_dct, outpath=None):
    '''
    Extract VOI mean values from PET image `impet` using image labels `imlabel`.
    Both can be dictionaries, file paths or Numpy arrays.
    They have to be aligned and have the same dimensions.
    If path (output) is given, the ROI masks will be saved to file(s).
    Arguments:
        - impet:    PET image as Numpy array
        - imlabel:  image of labels (integer values); the labels can come
                    from T1w-based parcellation or an atlas.
        - voi_dct:  dictionary of VOIs, with entries of labels creating
                    composite volumes
        - output:   
    '''


    # > assume none of the below are given 
    # > used only for saving ROI mask to file if requested
    affine, flip, trnsp = None, None, None

    if isinstance(impet, dict):
        im = impet['im']
        if 'affine' in impet:
            affine = impet['affine']
        if 'flip' in impet:
            flip =  impet['flip']
        if 'transpose' in impet:
            trnsp = impet['transpose']

    elif isinstance (impet, (str, PurePath)) and os.path.isfile(impet):
        imd = nimpa.getnii(impet, output='all')
        im = imd['im']
        flip = imd['flip']
        trnsp = imd['transpose']

    elif isinstance(impet, np.ndarray):
        im = impet


    if isinstance(imlabel, dict):
        lbls = imlabel['im']
        if 'affine' in imlabel and affine is None:
            affine = imlabel['affine']
        if 'flip' in imlabel and flip is None:
            flip =  imlabel['flip']
        if 'transpose' in imlabel and trnsp is None:
            trnsp = imlabel['transpose']

    elif isinstance (imlabel, (str, PurePath)) and os.path.isfile(imlabel):
        prd = nimpa.getnii(imlabel, output='all')
        lbls = prd['im']
        if affine is None:
            affine = prd['affine']
        if flip is None:
            flip =  prd['flip']
        if trnsp is None:
            trnsp = prd['transpose']

    elif isinstance(imlabel, np.ndarray):
        lbls = imlabel

    #----------------------------------------------
    # > output dictionary
    out = {}

    logging.debug('Extracting volumes of interest (VOIs):')
    for k, voi in enumerate(voi_dct):

        logging.info(f'  VOI: {v}')

        # > ROI mask
        rmsk = np.zeros(lbls.shape, dtype=bool)
        # > number of voxels in the ROI
        vxsum = 0
        # > voxel emission sum
        emsum = 0

        for ri in voi_dct[v]:
            logging.debug(f'   label{ri}')
            rmsk += np.equal(lbls, ri)

        if outpath is not None:
            nimpa.create_dir(outpath)
            fvoi = Path(outpath) / (voi+'_mask.nii.gz')
            nimpa.array2nii(
                rmsk.astype(np.int8),
                affine,
                fvoi,
                trnsp=(trnsp.index(0), trnsp.index(1), trnsp.index(2)),
                flip=flip)
        else:
            fvoi = None
        
        vxsum += np.sum(rmsk)
        emsum += np.sum(im[rmsk].astype(np.float64))


        out[voi] = {'vox_no':vxsum, 'sum':emsum, 'avg':emsum/vxsum, 'froi':froi, 'roimsk':rmsk}
    #----------------------------------------------

    return out
# ========================================================================================


# ========================================================================================
def preproc_suvr(pet_path, frames=None, outpath=None, fname=None):
    ''' Prepare the PET image for SUVr analysis.
        Arguments:
        - pet_path: path to the folder of DICOM images, or to the NIfTI file
        - outpath:  output folder path; if not given will assume the parent
                    folder of the input image
        - frames:   list of frames to be used for SUVr processing
    '''


    if not os.path.exists(pet_path):
        raise ValueError('The provided path does not exist')

    # > convert the path to Path object
    pet_path = Path(pet_path)


    #--------------------------------------
    # > sort out the output folder
    if outpath is None:
        outdir = pet_path.parent
    else:
        outdir = Path(outpath)

    petout = outdir / (pet_path.name+'_suvr')
    nimpa.create_dir(petout)

    if fname is None:
        fname = pet_path.name+'_suvr.nii.gz'
    elif not str(fname).endswith(nifti_ext[1]):
        fname += '.nii.gz'
    #--------------------------------------

    

    # > NIfTI case
    if pet_path.is_file() and str(pet_path).endswith(nifti_ext):
        logging.info('PET path exists and it is a NIfTI file')

        fpet_nii = pet_path

    # > DICOM case (if any file inside the folder is DICOM)
    elif pet_path.is_dir() and any([f.suffix in dicom_ext for f in pet_path.glob('*')]):

        # > get the NIfTi images from previous processing
        fpet_nii = list(petout.glob(pet_path.name + '*.nii*'))

        if not fpet_nii:
            run([dcm2niix.bin,
                 '-i', 'y',
                 '-v', 'n',
                 '-o', petout,
                 'f', '%f_%s',
                 pet_path])

        fpet_nii = list(petout.glob(pet_path.name + '*.nii*'))
        

        if not fpet_nii:
            raise ValueError('No SUVr NIfTI files found')
        elif len(fpet_nii)>1:
            raise ValueError('Too many SUVr NIfTI files found')
        else:
            fpet_nii = fpet_nii[0]

    # > read the dynamic image
    imdct = nimpa.getnii(fpet_nii, output='all')

    # > number of dynamic frames
    nfrm = imdct['hdr']['dim'][4]

    # > ensure that the frames exist in part of full dynamic image data
    if nfrm<max(frames):
        raise ValueError('The selected frames do not exist')

    logging.info(f'{nfrm} frames have been found in the dynamic image.')


    #------------------------------------------
    # > SUVr file path
    fsuvr = petout / fname

    #> check if the SUVr file already exists
    if not fsuvr.is_file():

        suvr_frm = np.sum(imdct['im'][frames, ...], axis=0)

        nimpa.array2nii(
                suvr_frm,
                imdct['affine'],
                fsuvr,
                trnsp = (imdct['transpose'].index(0),
                         imdct['transpose'].index(1),
                         imdct['transpose'].index(2)),
                flip = imdct['flip'])

        logging.info(f'Saved SUVr file image to: {fsuvr}')
    #------------------------------------------



    return dict(fpet_nii=fpet_nii, fpre_suvr=fsuvr)
# ========================================================================================









