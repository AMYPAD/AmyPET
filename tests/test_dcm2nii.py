from pathlib import Path

import pytest


@pytest.mark.timeout(30 * 60) # 30m
def test_dcm2nii(datain):
    dcmpth = Path(datain['mumapDCM'])
    assert not list(dcmpth.glob('test_dcm2nii_*.nii*'))
    dcm2nii = pytest.importorskip("amypet.dcm2nii")
    dcm2nii.run(datain['mumapDCM'], fcomment="test_dcm2nii_")
    assert list(dcmpth.glob('test_dcm2nii_*.nii*'))
    for f in dcmpth.glob('test_dcm2nii_*.nii*'):
        f.unlink()
