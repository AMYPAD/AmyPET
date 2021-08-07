from os import fspath
from pathlib import Path

import pytest


@pytest.fixture
def fimin():
    """test input"""
    res = (
        "/home/pawel/cs_nifty/DPUK_dl/py_test/TP0/"
        "DICOM_MPRAGE_20200226150442_15_N4bias.nii.gz"
    )
    res = (
        "/home/pawel/cs_nifty/DPUK_dl/py_test/"
        "NEW002_PETMR_V1_00015_MR_images_MPRAGE_MPRAGE_20200212145346_15.nii"
    )
    res = Path(res)
    if not res.is_file():
        pytest.skip("fimin not found")
    return res


def test_gif(fimin):
    gif = pytest.importorskip("amypad.gif")
    outpath = fimin.parent
    msg = (
        "make sure that SimpleITK is installed:" " conda install -c simpleitk simpleitk"
    )

    if "N4bias" not in fspath(fimin):
        nimpa = pytest.importorskip("niftypet.nimpa")
        # pytest.importorskip("SimpleITK", reason=msg)
        biascorr = nimpa.bias_field_correction(
            fspath(fimin), executable="sitk", outpath=fspath(outpath)
        )
        fin = biascorr["fim"]
    else:
        fin = fimin
        print(msg)

    gif.run(fin, outpath=outpath / "GIF")
