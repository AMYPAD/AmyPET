"""Convert DICOM folder to a NIfTI file

Usage:
  dcm2nii [options] <dcmpth>

Arguments:
  <dcmpth>  : Input folder containing DICOM files [default: DirChooser]

Options:
  --fcomment COMMENT  : Prefix to add to outputs
  --timestamp  : Whether to include a timestamp in the output filename
"""
import logging
from pathlib import Path

from niftypet import nimpa

log = logging.getLogger(__name__)


def run(dcmpth, fcomment="converted-from-DICOM_", timestamp=True):
    dcmpth = Path(dcmpth)
    assert dcmpth.is_dir()
    log.info("convert")
    res = nimpa.dcm2nii(dcmpth, fprefix=fcomment, timestamp=timestamp)
    log.debug("output file:%s", res)
    return res
