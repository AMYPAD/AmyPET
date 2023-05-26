"""Setting up AmyPET"""
__author__ = "Pawel J. Markiewicz"
__copyright__ = "Copyright 2023"


import importlib
import re
import platform
import os, sys
from textwrap import dedent

import logging
log = logging.getLogger(__name__)

def path_amypet_local():
    """ Get the path to the local (home) folder for AmyPET resources.

        If using `conda` put the resources in the folder with the environment name
    """

    if "CONDA_DEFAULT_ENV" in os.environ:
        try:
            env = re.findall(r"envs[/\\](.*)[/\\]bin[/\\]python", sys.executable)[0]
        except IndexError:
            env = os.environ["CONDA_DEFAULT_ENV"]
        log.info("install> conda environment found: {}".format(env))
    else:
        env = ""
    # create the path for the resources files according to the OS platform
    if platform.system() in ("Linux", "Darwin"):
        path_resources = os.path.expanduser("~")
    elif platform.system() == "Windows":
        path_resources = os.getenv("LOCALAPPDATA")
    else:
        raise ValueError("Unknown operating system: {}".format(platform.system()))
    path_resources = os.path.join(path_resources, ".amypet", env)

    return path_resources




def get_params(sys_append=True, reload=True):
    path_resources = path_amypet_local()
    if sys_append:
        if path_resources not in sys.path:
            sys.path.append(path_resources)
    try:
        import amypet_params
    except ImportError:
        log.error(
            dedent(
                """\
        --------------------------------------------------------------------------
        AmyPET parameter file <amypet_params.py> could not be imported.
        It should be in ~/.amypet/amypet_params.py (Linux) or
        in //Users//USERNAME//AppData//Local//amypet//amypet_params.py (Windows)
        but likely it does not exists.
        --------------------------------------------------------------------------"""
            )
        )
        raise
    else:
        return importlib.reload(amypet_params) if reload else amypet_params