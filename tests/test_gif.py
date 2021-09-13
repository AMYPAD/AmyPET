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


@pytest.mark.timeout(60 * 60 * 24)
def test_amypet_gif(dimin, caplog):
    amypet_gif = pytest.importorskip("scripts.amypet_gif")
    amypet_gif.main([fspath(dimin)])
