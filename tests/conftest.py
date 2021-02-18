import pytest


@pytest.fixture(scope="session")
def nvml():
    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        pytest.skip(str(exc))
    return pynvml
