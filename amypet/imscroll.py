"""Image viewer

Usage:
  imscroll [options] <search_dir>

Arguments:
  <search_dir>  : Input folder [default: DirChooser]

Options:
  --glob PATTERN  : File matching pattern [default: *.nii*]
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from miutil.fdio import nsort
from miutil.imio import imread
from miutil.plot import imscroll
from tqdm.contrib import tmap

log = logging.getLogger(__name__)


def run(search_dir, glob="*.nii*"):
    search_dir = Path(search_dir)
    assert search_dir.is_dir()
    fnames = list(map(str, search_dir.glob(glob)))
    log.info("found %d files", len(fnames))
    fnames = nsort(fnames)
    res = imscroll(list(tmap(imread, fnames)))
    plt.show()
    return res
