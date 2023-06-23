"""Centiloid pipeline: Process CL and dynamic amyloid PET data

Usage:
  centiloid [options] <inpath>

Arguments:
  <inpath>  : PET & MRI NIfTI directory [default: DirChooser]

Options:
  --outpath DIR  : Output directory
  --tracer TYPE  : PET Tracer; choices: {[default: pib],flt,fbp,fbb}
  --start TIME  : Uptake ratio (UR) start time [default: None:float]
  --end TIME  : Uptake ratio (UR) end time [default: None:float]
  --voxsz VOXELS  : [default: 2:int]
  --dynamic-break  : Use "coffee-break" model to fill measurement gaps
    [default: True]
  --bias-corr  : Perform bias correction
"""
__author__ = "Casper da Costa-Luis", "Pawel Markiewicz"
__copyright__ = "Copyright 2022-23"
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from .backend_centiloid import run as centiloid_run
from .align import align
from .dyn_tools import km_voi, km_img
from .preproc import get_t1, explore_indicom, convert2nii, rem_artefacts
from .proc import proc_vois
from .utils import params

log = logging.getLogger(__name__)
TRACERS = set(params['tracer_names'].keys())


def run(inpath, tracer='pib', start=None, end=None, dynamic_break=False, voxsz=2,
        bias_corr=True, outpath=None):
    """Just a stub"""
    inpath = Path(inpath)
    outpath = Path(outpath) if outpath else inpath.parent / f'amypet_output_{inpath.name}'
    ur_win_def = [start, end] if start and end else None # [t0, t1] - default: None

    ft1w = get_t1(inpath, params)  # structural/T1w image

    # processed & classify input data (e.g. auto identify UR frames)
    indat = explore_indicom(inpath, params, tracer=tracer, find_ur=True, ur_win_def=ur_win_def, outpath=outpath / 'DICOMs')
    niidat = convert2nii(indat, use_stored=True)
    # remove artefacts at the end of FoV
    niidat = rem_artefacts(niidat, params, artefact='endfov')
    # align pet frames for static/dynamic imaging
    aligned = align(niidat, params, use_stored=True)

    # calculate Centiloid (CL) quantification
    out_cl = centiloid_run(aligned['ur']['fur'], ft1w, params, stage='f', voxsz=voxsz, bias_corr=bias_corr, tracer=tracer, outpath=outpath / 'CL', use_stored=True)

    if False:
        # > Dynamic PET & kinetic analysis

        # The CL intermediate results are used to move the atlas & grey matter mask to PET space

        # get all the regional data with atlas and GM probabilistic mask
        dynvois = proc_vois(niidat, aligned, out_cl, outpath=niidat['outpath'].parent / 'DYN')
        # kinetic analysis for a chosen volume of interest (VOI), here the frontal lobe
        dkm = km_voi(dynvois,  voi='frontal', dynamic_break=dynamic_break,  model='srtmb_basis', weights='dur')

        # TODO: voxel-wise kinetic analysis
        dimkm = km_img(
            aligned['fpet'], dynvois['voi']['cerebellum']['avg'], dynvois['dt'], dynamic_break=dynamic_break,
            model='srtmb_basis',
            weights='dur',
            outpath=dynvois['outpath'])

        # # show the great matter mask and atlas
        # imscroll([dynvois['atlas_gm']['atlpet'], dynvois['atlas_gm']['gmpet']])
        # # plot time activity curve for the cerebellum
        # plot(np.mean(dynvois['dt'],axis=0), dynvois['voi']['cerebellum']['avg'])

        # figure paths (if run in full, with kinetic modelling)
        # if static only, then only CL output
        # if dynamic too, then the figure as well.
        return {'CL': next(iter(out_cl.values())), '_amypet_imscroll': (out_cl[next(iter(out_cl))]['fqc'], dkm['fig'])}

    return {'CL': next(iter(out_cl.values())), '_amypet_imscroll': out_cl[next(iter(out_cl))]['fqc']}
