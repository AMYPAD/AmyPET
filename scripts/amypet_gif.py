"""Usage:
  amypet_gif [options] <path>

Arguments:
  <path>  : directory. Layout: <path>/*/*/*MPRAGE*.nii*
"""
import logging
import os
import subprocess
from pathlib import Path
from textwrap import dedent

from argopt import argopt
from miutil import hasext
from niftypet import nimpa

from amypad import gif

log = logging.getLogger(__name__)


def filter_dir(path):
    return filter(lambda i: i.is_dir(), path)


def filter_nii(path):
    return list(filter(lambda i: i.is_file() and hasext(i, (".nii", "nii.gz")), path))


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = argopt(__doc__).parse_args(args=argv)
    gpth = Path(args.path)
    assert gpth.is_dir()

    for spth in filter_dir(gpth.iterdir()):
        for tpth in spth.iterdir():
            # check if MPRAGE DICOM folder is present
            for fldr in filter_dir(tpth.glob("*MPRAGE*")):
                # run dcm2niix to convert the DICOMs into NIfTI
                fnii = filter_nii(tpth.glob("*MPRAGE*"))
                if not len(fnii):
                    subprocess.run(
                        [nimpa.resources.DCM2NIIX]
                        + "-i y -v n -f %f_%s".split()
                        + ["-o", tpth, fldr]
                    )

            fn4b = list((tpth / "N4bias").glob("*N4bias*.nii*"))
            fgif = list((tpth / "GIF").glob("*Parcellation*.nii*"))

            if len(fn4b) < 1:
                fnii = filter_nii(tpth.glob("*MPRAGE*"))
                log.info("N4bias input:%s", fnii)
                biascorr = nimpa.bias_field_correction(
                    fnii[0], executable="sitk", outpath=tpth
                )
                try:
                    fingif = biascorr["fim"]
                except TypeError:
                    raise ImportError(
                        dedent(
                            """\
                            please install SimpleITK:
                            `conda install -c simpleitk simpleitk` or
                            `pip install SimpleITK`"""
                        )
                    )
            elif len(fn4b) == 1 and len(fgif) == 0:
                fingif = fn4b[0]
                log.debug("found N4-bias corrected:%s", fingif)
            else:
                continue

            log.info("running GIF on:%s", fingif)
            gif.run(fingif, outpath=os.path.join(tpth, "GIF"))


if __name__ == "__main__":
    main()
