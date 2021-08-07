from os import fspath
from pathlib import Path

import pytest


@pytest.fixture
def dyndir(datain):
    """
    TODO: use data like
    https://github.com/NiftyPET/NIPET/blob/master/tests/test_amyloid_pvc.py?
    """
    dyndir = datain / "amydyn"
    if dyndir.is_dir():
        return dyndir

    from functools import partial

    nipet = pytest.importorskip(
        "niftypet.nipet",
        reason="no existing data & no no NIPET so no way to reconstruct some",
    )

    # get all the constants and LUTs
    mMRpars = nipet.get_mmrparams()  # mMRpars["Cnt"]["VERBOSE"] = True
    # recognise the input data as much as possible
    classify_input = partial(nipet.classify_input, params=mMRpars)

    # ------------------------------------------------------
    # datain = classify_input("/store/downloads/1946/S00151_18715520/TP0")
    datain = classify_input("/data/amyloid_brain")
    # definition of dynamic frames for kinetic analysis
    frmdef = ["def", [4, 15], [8, 30], [9, 60], [2, 180], [8, 300]]
    # output path
    opth = fspath(Path(datain["corepath"]) / "output_dyn")
    # ------------------------------------------------------

    hst = nipet.mmrhist(datain, mMRpars)
    # offset for the time from which meaningful events are detected
    toff = nipet.lm.get_time_offset(hst)
    # dynamic frame timings
    frm_timings = nipet.lm.dynamic_timings(frmdef, offset=toff)
    nipet.lm.draw_frames(hst, frm_timings["timings"])
    # hardware mu-map
    muhdct = nipet.hdw_mumap(datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True)

    # object mu-map with alignment
    mupdct = nipet.align_mumap(
        datain,
        mMRpars,
        outpath=opth,
        store=True,
        hst=hst,
        itr=2,
        petopt="ac",
        fcomment="_mu",
        musrc="pct",
    )
    # object mu-map without alignment--straight from DICOM resampled to PET
    # muodct = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)

    nipet.mmrchain(
        datain,
        mMRpars,
        frames=frm_timings["timings"],
        mu_h=muhdct,
        mu_o=mupdct,  # muodct,
        itr=5,
        fwhm=0.0,
        outpath=opth,
        fcomment="_dyn",
        store_img=True,
        store_img_intrmd=True,
    )
    return Path(opth) / "PET" / "multiple-frames"


def test_rectrim(dyndir):
    rectrim = pytest.importorskip("amypad.rectrim")
    rectrim.run(dyndir)
