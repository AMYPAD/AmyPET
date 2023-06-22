"""Centiloid pipeline

Usage:
  centiloid [options] <inpath>

Arguments:
  <inpath>  : PET & MRI NIfTI directory [default: DirChooser]

Options:
  --outpath DIR  : Output directory
  --tracer TYPE  : PET Tracer; choices: {[default: pib],flt,fbp,fbb}
  --start TIME  : uptake ratio (UR) start time [default: None:float]
  --end TIME  : uptake ratio (UR) end time [default: None:float]
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
from .preproc import tracer_names
from .align import align_ur
from .ur_tools import preproc_ur

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
    ur_win_def = [start, end] if start and end else None # [t0, t1] - default: None

    # processed & classify input data (e.g. auto identify UR frames)
    from niftypet.nipet import explore_input
    indat = explore_input(inpath, tracer=tracer, ur_win_def=ur_win_def, outpath=outpath)

    # > find the UR-compatible acquisition and its index
    ur_find = [(i, a) for i, a in enumerate(indat['descr']) if 'ur' in a['acq']]

    if len(ur_find) > 1:
        raise IndexError('too many UR/static DICOM series detected: only one is accepted')
    elif len(ur_find) == 0:
        raise IndexError('could not identify any UR DICOM series in in the input data')
    else:
        # time-sorted data for UR
        ur_tdata = indat['series'][ur_find[0][0]]
        # data description with classification
        ur_descr = ur_find[0][1]

    # align the PET frames for UR/CL
    aligned = align_ur(ur_tdata, ur_descr, indat['outpath'])
    # preprocess the aligned PET into a single UR frame
    ur_preproc = preproc_ur(aligned['fpet'], outpath=aligned['outpath'])
    # calculate Centiloid (CL)
    out_cl = centiloid_run(ur_preproc['fstat'], ft1w, voxsz=voxsz, bias_corr=bias_corr,
                           tracer=tracer, outpath=aligned['outpath'] / 'CL')
    cl_dct = next(iter(out_cl.values()))

    return cl_dct
