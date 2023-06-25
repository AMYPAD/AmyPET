'''
Process CL and dynamic amyloid PET data
'''
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2023"

from pathlib import Path

import spm12
from niftypet import nimpa

import amypet
from amypet import backend_centiloid as centiloid
from amypet import params

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INPUT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# > input PET folder
input_fldr = Path('/sdata/PETpp/20170516_e00179_ANON179_analytics/')
outpath = input_fldr / 'amypet_processed_t'

#------------------------------
# > structural/T1w image
ft1w = amypet.get_t1(input_fldr, params)

if ft1w is None:
    raise ValueError('Could not find the necessary T1w DICOM or NIfTI images')
#------------------------------

petin = input_fldr / 'offline3D_TOFOSEM-PSF'
fpet = amypet.dicom2nifti(petin, outpath=outpath / 'petnii', remove_previous=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CL PIPELINE FOR PREPROCESSING
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
out_cl = centiloid.run(
    fpet,
    ft1w,
    params,
    stage='n',
    voxsz=2,
    bias_corr=True,
    outpath=outpath / 'norm',
    use_stored=True,
)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INTENSITY NORMALISATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fipet = amypet.iinorm(out_cl, fpet=fpet, refvoi=[17, 18], atlas='hammers', fcomment='amypet_inorm',
                      outpath=None, output_masks=True, apply_gmmask=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
