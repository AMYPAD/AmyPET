"""
Converted from Pipeline_Centiloid_BBRC.m
"""
from contextlib import contextmanager
from functools import lru_cache, wraps
from glob import glob
from os import path
from pkg_resources import resource_filename
import logging

from spm12 import ensure_spm
from tqdm.auto import tqdm, trange

log = logging.getLogger(__name__)
PATH_M = resource_filename(__name__, "")


@lru_cache()
@wraps(ensure_spm)
def get_matlab(**kwargs):
    eng = ensure_spm(**kwargs)
    log.debug("adding wrappers (%s) to MATLAB path", PATH_M)
    eng.addpath(PATH_M, nargout=0)
    return eng


@contextmanager
def tic(desc, leave=True, **kwargs):
    with tqdm(desc=desc, leave=leave, **kwargs) as _:
        yield


def run(
    dir_MRI="Projects/ALFA_PET",
    dir_PET="Projects/ALFA_PET",
    dir_RR="Atlas/CL_2mm",
    dir_quant="Projects/ALFA_PET/Quant_realigned",
):
    """
    Args:
        dir_MRI: MRI directory
        dir_PET: PET directory
        dir_RR: Reference regions ROIs directory
            (standard Centiloid RR from GAAIN Centioid website: 2mm, nifti)
        dir_quant: Quantification directory
    """
    s_PET_dir = glob(path.join(dir_PET, "*_PET.nii"))  # PET images list
    s_MRI_dir = glob(path.join(dir_MRI, "*_MRI.nii"))  # MR images lsit
    n_subj_PET = len(s_PET_dir)  # Number of PET images
    n_subj_MRI = len(s_MRI_dir)  # Number of MR images

    if n_subj_PET != n_subj_MRI:
        raise IndexError("Different number of PET and MR images")

    eng = get_matlab()
    dir_spm = path.dirname(eng.which("spm"))

    # TODO: in parallel
    for i_subj in trange(n_subj_PET):
        with tic("Step 0: Reorient PET subject"):
            d_PET = s_PET_dir[i_subj]
            eng.f_acpcReorientation(d_PET)

        with tic("Step 0: Reorient MRI subject"):
            d_MRI = s_MRI_dir[i_subj]
            eng.f_acpcReorientation(d_MRI)

        with tic("Step 1: CorregisterEstimate"):
            eng.f_1CorregisterEstimate(d_MRI, dir_spm)
        # Check Reg

        with tic("Step 2: CorregisterEstimate"):
            eng.f_2CorregisterEstimate(d_MRI, d_PET)
        # Check Reg

        with tic("Step 3: Segment"):
            eng.f_3Segment(d_MRI, dir_spm)

        with tic("Step 4: Normalise"):
            d_file_norm = path.join(dir_MRI, "y_" + path.basename(s_MRI_dir[i_subj]))
            eng.f_4Normalise(d_file_norm, d_MRI, d_PET)

    s_PET = glob(path.join(dir_PET, "w*PET.nii"))  # MODIFY
    eng.f_Quant_centiloid(s_PET, dir_RR, dir_quant)
