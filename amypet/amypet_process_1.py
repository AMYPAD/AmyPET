'''
Upload mMR data to XNAT (central-dupk)
'''
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2022"



import numpy as np
import os, sys, glob, shutil
from pathlib import Path
import urllib

from niftypet import nimpa
import spm12
import amypet
from amypet import backend_centiloid as centiloid

import logging as log

log.basicConfig(level=log.WARNING, format=nimpa.LOG_FORMAT)

pttrn_t1 = ['mprage', 't1', 't1w']



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INPUT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# > input PET folder
#input_fldr = Path('/data/AMYPET/PNHS_test_data/FBB1')
#input_fldr = Path('/data/AMYPET/PNHS_test_data/FBB1/DYN_PET_AC')
input_fldr = Path('/home/pawel/data/AMYPET/PNHS_test_data/FBB1/ST_PET_AC')

# > find an MR T1w image
ft1w = None
for f in input_fldr.iterdir():
    if f.is_file() and f.name.endswith(('.nii','.nii.gz')) and any([p in f.name.lower() for p in pttrn_t1]):
        ft1w = f
        break        

if ft1w is None:
    raise ValueError('Could not find the necessary T1w NIfTI image')

# > SUVr window def
suvr_win_def=[5400,6600]

tracer = 'fbb' # 'pib', 'flute', 'fbp'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IDENTIFY SUVR STATIC DATA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
outpath = input_fldr.parent/('amypet_output_'+input_fldr.name) 

# > processed the input data and classify it (e.g., automatically identify SUVr frames)
indat = amypet.explore_input(input_fldr, tracer=tracer, suvr_win_def=suvr_win_def, outpath=outpath)

# > find the SUVr-compatible acquisition and its index
suvr_find = [(i,a) for i,a in enumerate(indat['descr']) if 'suvr' in a['acq']]

if len(suvr_find)>1:
    raise ValueError('too many SUVr/static DICOM series detected: only one is accepted')
elif len(suvr_find)==0:
    raise ValueError('could not identify any SUVr DICOM series in in the input data')
else:

    # > time-sorted data for SUVr
    suvr_tdata = indat['series'][suvr_find[0][0]]

    # > data description with classification
    suvr_descr = suvr_find[0][1]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ALIGN PET FRAMES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# > align the PET frames for SUVr/CL
aligned = amypet.align_suvr(suvr_tdata, suvr_descr, indat['outpath'])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROCESS PET FOR SUVR AND CL
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# > preprocess the aligned PET into a single SUVr frame
suvr_preproc = amypet.preproc_suvr(aligned['fpet'], outpath=aligned['outpath'])

# > calculate Centiloid (CL)
out_cl = centiloid.run(
    suvr_preproc['fstat'],
    ft1w,
    voxsz=2,
    bias_corr=True,
    tracer=tracer,
    outpath=aligned['outpath']/'CL')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PROCESS FOR NATIVE PET ANALYSIS
# Use GM SPM segmentation to refine the MNI atlas transformed to
# the native PET space
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cl_dct = out_cl[next(iter(out_cl))]

native_pet = amypet.native_proc(cl_dct, atlas='aal', res=1, outpath=aligned['outpath'], refvoi_idx=range(91,113))


