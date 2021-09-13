from os import fspath
from pathlib import Path

import pytest

nipet = pytest.importorskip("niftypet.nipet")


@pytest.fixture
def dyndir(datain, mMRpars):
    # definition of dynamic frames for kinetic analysis
    frmdef = ["def", [4, 15], [8, 30], [9, 60], [2, 180], [8, 300]]
    # output path
    opth = fspath(Path(datain["corepath"]) / "output_dyn")

    res = Path(opth) / "PET" / "multiple-frames"  # datain / "amydyn"
    if res.is_dir():
        return res

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
