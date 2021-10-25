"""Parcellation

Usage:
  GIF [options] <fimin>

Arguments:
  <fimin>  : NIfTI image input file (T1w MR image) [default: FileChooser]

Options:
  --outpath DIR  : where to write outputs of GIF
  --gif DIR  : GIF directory (containing `bin/` & `db/`)
               (default: ${PATHTOOLS}/GIF_AMYPET) [default: DirChooser]
"""
import errno
import subprocess
from os import fspath, getenv
from pathlib import Path

from .utils import cpu_count


def run(fimin, outpath=None, gif=None):
    if not gif and getenv("PATHTOOLS"):
        gif = Path(getenv("PATHTOOLS")) / "GIF_AMYPET"
    gif = Path(gif)
    if not all(i.is_dir() for i in [gif, gif / "bin", gif / "db"]):
        raise FileNotFoundError("GIF required")

    fimin = Path(fimin)
    if not fimin.is_file() or "nii" not in fspath(fimin):
        raise IOError(errno.ENOENT, "Nifty input file required", fimin)

    if outpath is None:
        opth = fimin.parent / "out"
    else:
        opth = Path(outpath)
    opth.mkdir(mode=0o775, parents=True, exist_ok=True)
    gifresults = subprocess.run([
        fspath(gif / "bin" / "seg_GIF"), '-in', fimin, '-db',
        fspath(gif / "db" / "db.xml"), '-v', "1", '-regNMI', '-segPT', "0.1", '-out',
        fspath(opth), '-temper', "0.05", '-lncc_ker', "-4", '-omp',
        str(cpu_count()), '-regBE', "0.001", '-regJL', "0.00005"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    (opth / "out.log").write_bytes(gifresults.stdout)
    (opth / "err.log").write_bytes(gifresults.stderr)

    return gifresults
