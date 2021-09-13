from os import getenv
from pathlib import Path

import pytest

HOME = Path(getenv("DATA_ROOT", "~")).expanduser()


@pytest.fixture(scope="session")
def nvml():
    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        pytest.skip(str(exc))
    return pynvml


@pytest.fixture(scope="session")
def dimin():
    trt = HOME / "amypad" / "DPUK" / "TRT"
    if not trt.is_dir():
        pytest.skip(
            f"""Cannot find amypad/DPUK/TRT in
${{DATA_ROOT:-~}} ({HOME}).
"""
        )
    return trt


@pytest.fixture(scope="session")
def fimin(dimin):
    mprage = (
        dimin
        / "NEW002_ODE_S02442"
        / "TP0"
        / "NEW002_PETMR_V1_00015_MR_images_MPRAGE_q-_MPRAGE_20200212145346_15.nii"
    )
    if not mprage.is_file():
        pytest.skip("fimin not found")
    return mprage
