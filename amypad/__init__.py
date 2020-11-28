# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"

try:
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        __licence__ = get_distribution("amypad").get_metadata("LICENCE.md")
    except DistributionNotFound:
        raise ImportError
except ImportError:
    from os import path

    try:
        __licence__ = open(
            path.join(path.dirname(path.dirname(__file__)), "LICENCE.md")
        ).read()
    except FileNotFoundError:
        __licence__ = "Apache-2.0"
