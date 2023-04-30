'''
Static frames processing tools for AmyPET 
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging as log
import os, shutil
from pathlib import Path, PurePath
from subprocess import run

import numpy as np
from niftypet import nimpa
import spm12



log.basicConfig(level=log.WARNING, format=nimpa.LOG_FORMAT)



def timing_dyn(niidat):
    '''
    Get the frame timings from the NIfTI series
    '''

    t = []
    for dct in niidat['descr']:
        t+=dct['timings']
    t.sort()
    # > time mid points
    tm = [(tt[0]+tt[1])/2 for tt in t]

    # > convert the timings to NiftyPAD format
    dt = np.array(t).T

    return dict(timings=t, niftypad=dt, t_mid_points=tm)






