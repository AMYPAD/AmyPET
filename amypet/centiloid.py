"""Centiloid pipeline

Usage:
  centiloid [options] <inpath>

Arguments:
  <inpath>  : PET & MRI NIfTI directory [default: DirChooser]

Options:
  --outpath DIR  : Output directory
  --tracer TYPE  : PET Tracer; choices: {[default: pib],flt,fbp,fbb}
  --start TIME  : SUVr start time [default: None:float]
  --end TIME  : SUVr end time [default: None:float]
  --dynamic-analysis  : whether to do dynamic analysis
  --voxsz VOXELS  : [default: 2:int]
  --bias-corr  : Perform bias correction
"""
__author__ = ("Casper da Costa-Luis", "Pawel Markiewicz")
__copyright__ = "Copyright 2022"
import logging
import re
from pathlib import Path

from .backend_centiloid import run as centiloid_run
from .preproc import align_suvr, explore_input, tracer_names
from .suvr_tools import preproc_suvr

log = logging.getLogger(__name__)
# TRACERS = ('pib', 'flt', 'fbp', 'fbb')
TRACERS = tracer_names.keys()


def run(inpath, tracer='pib', start=None, end=None, dynamic_analysis=False, voxsz=2,
        bias_corr=True, outpath=None):
    """Just a stub"""
    inpath = Path(inpath)
    outpath = Path(outpath) if outpath else inpath.parent / f'amypet_output_{inpath.name}'
    if dynamic_analysis:
        print("Warning: dynamic_analysis ignored")

    # find an MR T1w image
    ft1w = None
    for f in inpath.iterdir():
        if f.is_file() and re.search(r"(mprage|t1|t1w).*\.nii(\.gz)?$", f.name, flags=re.I):
            ft1w = f
            break
    else:
        print(FileNotFoundError('Could not find the necessary T1w NIfTI image'))
    suvr_win_def = [start, end] if start and end else None # [t0, t1] - default: None

    # processed & classify input data (e.g. auto identify SUVr frames)
    indat = explore_input(inpath, tracer=tracer, suvr_win_def=suvr_win_def, outpath=outpath)

    # > find the SUVr-compatible acquisition and its index
    suvr_find = [(i, a) for i, a in enumerate(indat['descr']) if 'suvr' in a['acq']]

    if len(suvr_find) > 1:
        raise IndexError('too many SUVr/static DICOM series detected: only one is accepted')
    elif len(suvr_find) == 0:
        raise IndexError('could not identify any SUVr DICOM series in in the input data')
    else:
        # time-sorted data for SUVr
        suvr_tdata = indat['series'][suvr_find[0][0]]
        # data description with classification
        suvr_descr = suvr_find[0][1]

    # align the PET frames for SUVr/CL
    aligned = align_suvr(suvr_tdata, suvr_descr, indat['outpath'])
    # preprocess the aligned PET into a single SUVr frame
    suvr_preproc = preproc_suvr(aligned['fpet'], outpath=aligned['outpath'])
    # calculate Centiloid (CL)
    out_cl = centiloid_run(suvr_preproc['fstat'], ft1w, voxsz=voxsz, bias_corr=bias_corr,
                           tracer=tracer, outpath=aligned['outpath'] / 'CL')
    return next(iter(out_cl.values()))
