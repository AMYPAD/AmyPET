# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"

__licence__ = "MPL-2.0"

from .align import *
from .dyn_tools import *
from .preproc import *
from .proc import *
from .ur_tools import *
from .utils import *
