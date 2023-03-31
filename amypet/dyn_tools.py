'''
Static frames processing tools for AmyPET 
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging as log
import os
from pathlib import Path, PurePath
from subprocess import run

import numpy as np
from niftypet import nimpa
import matlab as ml


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








def atl2pet(frefpet, fatl, cldct, outpath=None):
    ''' 
    Atlas and GM from the centiloid (CL) pipeline to the reference
    PET space.
    Arguments:
    - frefpet:  the file path of the reference PET image 
    - fatl:     the file path of the atlas in MNI space
    - cldct:    the CL output dictionary
    '''

    # > output path
    if outpath is None:
        opth = Path(cl_dct['opth']).parent.parent
    else:
        opth = Path(outpath)
    nimpa.create_dir(opth)


    # > decipher the CL dictionary
    if len(cldct)==1:
        cl_dct = cldct[next(iter(cldct))]
    elif 'norm' in cldct:
        cl_dct = cldct
    else:
        raise ValueError('unrecognised CL dictionary')

    # > read the PET image
    petdct = nimpa.getnii(frefpet, output='all')

    # > SPM bounding box of the PET image
    bbox = spm12.get_bbox(petdct)


    # # > get the affine PET->MR
    # if isinstance(affine, (Path, PurePath)) or isinstance(affine, str):
    #     aff = np.loadtxt(affine)
    # elif isinstance(affine, np.array):
    #     aff = np.array(affine)

    # > get the inverse affine transform to PET native space
    M = np.linalg.inv(cl_dct['reg2']['affine'])
    Mm = ml.double(M.tolist())

    # > copy the inverse definitions to be modified with affine to native PET space
    fmod = shutil.copyfile(cl_dct['norm']['invdef'],
                           opth/(Path(cl_dct['norm']['invdef']).name.split('.')[0] + '_2nat.nii'))
    eng = spm12.ensure_spm('')
    eng.amypad_coreg_modify_affine(str(fmod), Mm)

    # > unzip the atlas and transform it to PET space
    fniiatl = nimpa.nii_ugzip(fatl, outpath=opth)

    # > inverse transform the atlas to PET space
    finvatl = spm12.normw_spm(str(fmod), [fniiatl + ',1'], voxsz=petdct['voxsize'], intrp=0., bbox=bbox,
                              outpath=str(opth))[0]

    # > remove the uncompressed input atlas after transforming it
    os.remove(fniiatl)

    # > GM mask
    fgmpet = spm12.resample_spm(frefpet, cl_dct['norm']['c1'], M, intrp=1.0, outpath=opth,
                                pickname='flo', fcomment='_GM_in_PET', del_ref_uncmpr=True,
                                del_flo_uncmpr=True, del_out_uncmpr=True)


    # > remove NaNs
    atl = nimpa.getnii(finvatl)
    atl[np.isnan(atl)] = 0
    gm = nimpa.getnii(fgmpet)
    gm[np.isnan(gm)] = 0


    return dict(fatlpet=finvatl, fgmpet=fgmpet, atlmap=atl, gmap=gm, outpath=opth)
