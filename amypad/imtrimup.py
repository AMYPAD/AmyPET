"""Trimming and upscaling

Usage:
  imtrimup [options] <dyndir>

Arguments:
  <dyndir>  : Input folder containing dynamic scans [default: DirChooser]

Options:
  --glob PATTERN  : File matching pattern [default: *.nii*]
  --scale FACTOR  : Sampling scale factor [default: 2:int]
  --refim PATH  : Reference image [default: FileChooser]
  --fcomment COMMENT  : Prefix to add to outputs
  --no-memlim  : Whether to use more memory (faster)
  --store-img-intrmd  : Whether to write output image files
  --store-img  : Whether to write output image sum file
"""
import logging
from pathlib import Path

from niftypet import nimpa

log = logging.getLogger(__name__)


def run(
    dyndir,
    glob="*.nii*",
    scale=2,
    refim="",
    fcomment="",
    no_memlim=False,
    store_img_intrmd=False,
    store_img=False,
):
    dyndir = Path(dyndir)
    assert dyndir.is_dir()
    # get the file data
    fdyns = list(map(str, dyndir.glob(glob)))
    log.info("found %d files", len(fdyns))
    log.info("sort the dynamic images and output the sorted file names and numpy array")
    imdyn = nimpa.niisort(fdyns)
    log.info("trim & upscale")
    res = nimpa.imtrimup(imdyn["files"], scale=scale, memlim=not no_memlim, refim=refim or "",
                         fcomment=fcomment or "", store_img_intrmd=store_img_intrmd,
                         store_img=store_img)
    log.debug("trimmed and scaled images, shape %r", res['im'].shape)
    return res
