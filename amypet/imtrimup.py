import logging
from pathlib import Path

from niftypet import nimpa

log = logging.getLogger(__name__)


def run(
    dyndir,
    glob="*.nii*",
    scale=2,
    memlim=True,
    refim="",
    fcomment="",
    store_img_intrmd=False,
    store_img=False,
):
    """Trimming and upscaling
    Args:
      dyndir: folder containing dynamic scans
      glob: matching pattern for folder contents
    """
    dyndir = Path(dyndir)
    assert dyndir.is_dir()
    # get the file data
    fdyns = list(map(str, dyndir.glob(glob)))
    log.info("sort the dynamic images and output the sorted file names and numpy array")
    imdyn = nimpa.niisort(fdyns)
    log.info("trim & upscale")
    # NB: the first row is 'blank' due to the offset. TODO: check
    dtrm = nimpa.imtrimup(
        imdyn["files"][1:],
        scale=scale,
        memlim=memlim,
        refim=refim,
        fcomment=fcomment,
        store_img_intrmd=store_img_intrmd,
        store_img=store_img,
    )
    log.debug("trimmed and scaled images")
    res = dtrm["fimi"]
    log.debug("%r", res["im"].shape)
    return res
