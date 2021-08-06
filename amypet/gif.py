import multiprocessing
import os
from subprocess import run

from niftypet import nimpa

nthrds = multiprocessing.cpu_count()


# ====================================
def rungif(fimin, gifpath=None, outpath=None):
    """
    fimin   - NIfTI image input file (T1w MR image)
    gifpath - path to the GIF folders with executables and database

    """

    if gifpath is None or not os.path.exists(gifpath):
        raise ValueError("wrong path to GIF.")

    if not os.path.isfile(fimin) or "nii" not in fimin:
        raise ValueError("incorrect input image file.")

    gifexe = os.path.join(gifpath, "bin", "seg_GIF")
    gifdb = os.path.join(gifpath, "db", "db.xml")

    # ---------------------------------------------
    # > create outputs
    if outpath is None:
        opth = os.path.join(os.path.dirname(fimin), "out")
    else:
        opth = outpath
    nimpa.create_dir(opth)
    logerr = os.path.join(opth, "err.log")
    logout = os.path.join(opth, "out.log")
    # ---------------------------------------------

    # ---------------------------------------------
    gifresults = run(
        [
            gifexe,
            "-in",
            fimin,
            "-db",
            gifdb,
            "-v",
            "1",
            "-regNMI",
            "-segPT",
            "0.1",
            "-out",
            opth,
            "-temper",
            "0.05",
            "-lncc_ker",
            "-4",
            "-omp",
            str(nthrds),
            "-regBE",
            "0.001",
            "-regJL",
            "0.00005",
        ],
        capture_output=True,
    )
    # ---------------------------------------------

    # ---------------------------------------------
    # > output logs
    with open(logerr, "wb") as f:
        f.write(gifresults.stderr)
    with open(logout, "wb") as f:
        f.write(gifresults.stdout)
    # ---------------------------------------------

    return gifresults


# ====================================


# test input
fimin = (
    "/home/pawel/cs_nifty/DPUK_dl/py_test/TP0/"
    "DICOM_MPRAGE_20200226150442_15_N4bias.nii.gz"
)
fimin = (
    "/home/pawel/cs_nifty/DPUK_dl/py_test/"
    "NEW002_PETMR_V1_00015_MR_images_MPRAGE_MPRAGE_20200212145346_15.nii"
)

outpath = os.path.dirname(fimin)


if "N4bias" not in fimin:
    # import SimpleITK as sitk
    biascorr = nimpa.bias_field_correction(fimin, executable="sitk", outpath=outpath)
    fingif = biascorr["fim"]
else:
    fingif = fimin

    print("make sure that SimpleITK is installed: conda install -c simpleitk simpleitk")

rungif(
    fingif, gifpath="/home/pawel/AMYPET/GIF2BBRC", outpath=os.path.join(outpath, "GIF")
)
