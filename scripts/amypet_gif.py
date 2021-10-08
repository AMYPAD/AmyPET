"""Usage:
  amypet_gif [options] <path>

Options:
  -O, --overwrite-existing  : whether to run even if outputs already exist

Arguments:
  <path>  : directory. Layout: <path>/*/*/*MPRAGE*.nii*
"""
import logging
import os
import subprocess
from pathlib import Path
from textwrap import dedent

from argopt import argopt
from miutil import hasext, nsort
from niftypet import nimpa

from amypad import gif

log = logging.getLogger(__name__)


def filter_dir(path):
    return filter(lambda i: i.is_dir(), path)


def filter_nii(path):
    return list(filter(lambda i: i.is_file() and hasext(i, ("nii", "nii.gz")), path))


def main(argv=None):
    args = argopt(__doc__).parse_args(args=argv)
    gpth = Path(args.path)
    assert gpth.is_dir()

    for spth in filter_dir(gpth.iterdir()):
        for tpth in spth.iterdir():
            for fldr in filter_dir(tpth.glob("*MPRAGE*")):
                fnii = filter_nii(tpth.glob("*MPRAGE*"))
                if not len(fnii):
                    # convert DICOMs to NIfTI
                    subprocess.run([nimpa.resources.DCM2NIIX] + "-i y -v n -f %f_%s".split() +
                                   ["-o", tpth, fldr])

            fn4b = list((tpth / "N4bias").glob("*N4bias*.nii*"))
            fgif = list((tpth / "GIF").glob("*Parcellation*.nii*"))

            if len(fn4b) < 1:
                fnii = filter_nii(tpth.glob("*MPRAGE*"))
                log.info("N4bias input:%s", fnii)
                biascorr = nimpa.bias_field_correction(fnii[0], executable="sitk", outpath=tpth)
                try:
                    fingif = biascorr["fim"]
                except TypeError:
                    raise ImportError(
                        dedent("""\
                            please install SimpleITK:
                            `conda install -c simpleitk simpleitk` or
                            `pip install SimpleITK`"""))
            else:
                if len(fn4b) > 1:
                    log.warning("%d inputs found, selecting latest")
                fingif = nsort(list(map(str, fn4b)))[-1]
                log.debug("found N4-bias corrected:%s", fingif)
                if fgif and not args.overwrite_existing:
                    log.warning("skipping")
                    continue

            log.info("running GIF on:%s", fingif)
            gif.run(fingif, outpath=os.path.join(tpth, "GIF"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
