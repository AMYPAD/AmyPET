from pathlib import Path

import pytest


@pytest.fixture
def dyndir(datain, mMRpars):
    # output path
    opth = str(Path(datain["corepath"]).parent / "amypad" / "dyndir")

    res = Path(opth) / "PET" / "multiple-frames"
    if res.is_dir() and len(list(res.glob('*.nii*'))) > 1:
        return res

    nipet = pytest.importorskip("niftypet.nipet")
    hst = nipet.mmrhist(datain, mMRpars)
    # offset for the time from which meaningful events are detected
    toff = nipet.lm.get_time_offset(hst)
    # dynamic frame timings
    # frmdef = ["def", [4, 15], [8, 30], [9, 60], [2, 180], [8, 300]]
    frmdef = ["def", [4, (hst['t1'] - hst['t0'] - toff) / 4]]
    frm_timings = nipet.lm.dynamic_timings(frmdef, offset=toff)
    # nipet.lm.draw_frames(hst, frm_timings["timings"]) # plot frames
    # hardware mu-map
    muhdct = nipet.hdw_mumap(datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True)

    # object mu-map with alignment
    mupdct = nipet.align_mumap(
        datain,
        mMRpars,
        outpath=opth,
        store=True,
        use_stored=True,
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
        mu_o=mupdct,
        itr=5,
        fwhm=0.0,
        outpath=opth,
        fcomment="_dyn",
        store_img=True,
        store_img_intrmd=True,
    )
    return Path(opth) / "PET" / "multiple-frames"


@pytest.mark.timeout(30 * 60) # 30m
def test_imtrimup(dyndir):
    imtrimup = pytest.importorskip("amypet.imtrimup")
    imtrimup.run(dyndir, glob='*_frm?_t*.nii*')
