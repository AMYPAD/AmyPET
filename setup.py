from setuptools import setup
import os, sys, shutil

path_current = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(path_current,'amypet'))
from amypet_config import path_amypet_local


path_resources = path_amypet_local()


# > check if the local folder for AmyPET exists; if not create one.
if not os.path.exists(path_resources):
    os.makedirs(path_resources)

if not os.path.isfile(os.path.join(path_resources, "amypet_params.py")):
    if os.path.isfile(os.path.join(path_current, "params", "amypet_params.py")):
        shutil.copyfile(
            os.path.join(path_current, "params", "amypet_params.py"),
            os.path.join(path_resources, "amypet_params.py"),
        )


setup(use_scm_version=True)
