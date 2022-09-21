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
"""
__author__ = "Casper da Costa-Luis"
__copyright__ = "Copyright 2022"
import logging

from .backend_centiloid import run as centiloid_run

log = logging.getLogger(__name__)
TRACERS = ('pib', 'flt', 'fbp', 'fbb')


def run(inpath, tracer='pib', start=None, end=None, dynamic_analysis=False, outpath=None):
    """Just a stub"""
    # probably want to use `centiloid_run()`?
    return {
        'inpath': inpath, 'tracer': tracer, 'start': start, 'end': end,
        'dynamic': dynamic_analysis, 'outpath': outpath}
