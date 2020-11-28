"""
Converted from Pipeline_Centiloid_BBRC.m
"""
import logging
from contextlib import contextmanager
from functools import lru_cache, wraps
from glob import glob
from os import path

from pkg_resources import resource_filename
from tqdm.auto import tqdm
from tqdm.contrib import tmap, tzip

from miutil.imio import nii
from spm12 import ensure_spm

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


def gunzip(nii_gz):
    """`nii_ugzip` but skips if already existing"""
    fdir, base = nii.file_parts(nii_gz)[:2]
    fout = path.join(fdir, base, ".nii")
    return fout if path.exists(fout) else nii.nii_ugzip(nii_gz)


def run(
    dir_MRI="data/ALFA_PET",
    dir_PET="data/ALFA_PET",
    dir_RR="data/Atlas/CL_2mm",
    dir_quant="data/ALFA_PET/Quant_realigned",
    glob_PET="*_PET.nii.gz",
    glob_MRI="*_MRI.nii.gz",
):
    """
    Args:
        dir_MRI: MRI directory
        dir_PET: PET directory
        dir_RR: Reference regions ROIs directory
            (standard Centiloid RR from GAAIN Centioid website: 2mm, nifti)
        dir_quant: Quantification directory
    """
    # PET & MR images lists
    s_PET_dir = list(tmap(gunzip, glob(path.join(dir_PET, glob_PET)), leave=False))
    s_MRI_dir = list(tmap(gunzip, glob(path.join(dir_MRI, glob_MRI)), leave=False))
    if len(s_PET_dir) != len(s_MRI_dir):
        raise IndexError("Different number of PET and MR images")

    eng = get_matlab()
    dir_spm = path.dirname(eng.which("spm"))

    # TODO: in parallel
    for d_PET, d_MRI in tzip(s_PET_dir, s_MRI_dir):
        with tic("Step 0: Reorient PET subject"):
            eng.f_acpcReorientation(d_PET, nargout=0)

        with tic("Step 0: Reorient MRI subject"):
            eng.f_acpcReorientation(d_MRI, nargout=0)

        with tic("Step 1: CorregisterEstimate"):
            eng.f_1CorregisterEstimate(d_MRI, dir_spm, nargout=0)
        # Check Reg

        with tic("Step 2: CorregisterEstimate"):
            eng.f_2CorregisterEstimate(d_MRI, d_PET, nargout=0)
        # Check Reg

        with tic("Step 3: Segment"):
            eng.f_3Segment(d_MRI, dir_spm, nargout=0)

        with tic("Step 4: Normalise"):
            d_file_norm = path.join(dir_MRI, "y_" + path.basename(d_MRI))
            eng.f_4Normalise(d_file_norm, d_MRI, d_PET, nargout=0)

    s_PET = glob(path.join(dir_PET, "w*PET.nii"))  # MODIFY
    return eng.f_Quant_centiloid(s_PET, dir_RR, dir_quant)
