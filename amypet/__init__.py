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

from .align import *  # noqa, yapf: disable
from .dyn_tools import *  # noqa, yapf: disable
from .preproc import *  # noqa, yapf: disable
from .proc import *  # noqa, yapf: disable
from .ur_tools import *  # noqa, yapf: disable
from .utils import *  # noqa, yapf: disable
