import logging
from os import fspath

import pytest


def test_gif(fimin):
    gif = pytest.importorskip("amypad.gif")
    outpath = fimin.parent

    if "N4bias" not in fspath(fimin):
        nimpa = pytest.importorskip("niftypet.nimpa")
        pytest.importorskip(
            "SimpleITK",
            reason="conda install -c simpleitk simpleitk or pip install SimpleITK",
        )
        biascorr = nimpa.bias_field_correction(
            fspath(fimin), executable="sitk", outpath=fspath(outpath)
        )
        fin = biascorr["fim"]
    else:
        fin = fimin

    gif.run(fin, outpath=outpath / "GIF")


def test_amypet_gif(dimin, caplog):
    amypet_gif = pytest.importorskip("scripts.amypet_gif")
    with caplog.at_level(logging.DEBUG):
        amypet_gif.main([fspath(dimin)])
        assert not caplog.record_tuples
