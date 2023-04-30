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
import nibabel as nib
import matplotlib.pyplot as plt
from niftypet import nimpa
import spm12

#--- NiftyPAD ---
from niftypad import basis
from niftypad.kt import dt2mft, dt_fill_gaps, dt2tdur
from niftypad.image_process.parametric_image import image_to_parametric
from niftypad.models import get_model_inputs
from niftypad.tac import TAC, Ref
#---


log.basicConfig(level=log.WARNING, format=nimpa.LOG_FORMAT)


# =====================================================================
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





# =====================================================================
def km_voi(
    dynvois,
    voi=None,
    dynamic_break=False,
    model='srtmb_basis',
    beta_lim=None,
    n_beta=100,
    k2p=None,
    weights='dur',
    outpath=None):

    '''
    Run kinetic modelling (KM) using NiftyPAD
    Arguments:
    dynvois:    dictionary of dynamic VOIs based on the chosen atlas;
                also contains the frame timings and used atlases/masks.
    model:      model selection, one of the following: 
                'srtmb_basis', 'srtm', 'srtmb_k2p_basis', etc
    dynamic_break: if True, the so called coffee-break modelling is used
                to fill the gaps when measurement was not taken.
    beta_lim:   KM input parameter (see NiftyPAD).
    n_beta:     basis number (see NiftyPAD).
    k2p:        KM input parameter (see NiftyPAD)
    weights:    KM weights for each dynamic frame; if set to 'dur'
                the duration of each frame will be used for setting up
                the weights; if set to None, no weights are used.

    '''

    #----------------------------------
    # > sorting output folder
    if outpath is None:
        opth = dynvois['outpath']
    else:
        opth = Path(outpath)
        nimpa.create_dir(opth)
    #----------------------------------

    # > start/stop times for each frame
    dt = dynvois['dt']

    # > reference region VOI, with interpolation
    ref = Ref(dynvois['voi']['cerebellum']['avg'], dt)
    ref.interp_1cubic()

    # > target TAC of selected VOI
    tac = TAC(dynvois['voi'][voi]['avg'], dt)


    #----- KM parameters -----
    if beta_lim is None:
        beta_lim = [0.01/60, 0.5/60]

    if weights is None:
        w = None
    elif weights=='dur':
        w = dt2tdur(dt)
        w = w / np.amax(w)
    else:
        raise ValueError('currently unrecognised weights for dynamic frames')
    #-------------------------


    b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=w, k2p=k2p)
    user_inputs = {
        'dt':dt,
        'inputf1':ref.inputf1cubic,
        'w':w,
        'k2p':k2p,
        'beta_lim':beta_lim,
        'n_beta':n_beta,
        'b': b}

    model_inputs = get_model_inputs(user_inputs, model)
    tac.run_model(model, model_inputs)
    km_res1 = tac.km_results

    if dynamic_break:
        dt_gaps = dt_fill_gaps(dt)
        # >>> TODO: check what or if none weights to use here:
        b_gaps = basis.make_basis(ref.inputf1cubic, dt_gaps, beta_lim=beta_lim, n_beta=n_beta,
                                       w=None, k2p=k2p)
        inputf1_dt_gaps = ref.inputf1cubic, dt_gaps
        user_inputs_gaps = {'b': b_gaps, 'inputf1_dt': inputf1_dt_gaps}
    
        tac.km_results.update(user_inputs_gaps)
        km_res1 = tac.km_results
        model_km_inputs = get_model_inputs(km_res1, model + '_para2tac')
        tac.run_model_para2tac(model + '_para2tac', model_km_inputs)


    r1 = km_res1['r1']
    k2 = km_res1['k2']
    bp = km_res1['bp']

    fig, ax = plt.subplots(2, 1, figsize=(9, 12))

    ax[0].plot(ref.input_interp_1, label='Int. ref. TAC')
    ax[0].plot(np.mean(dt,axis=0), ref.tac, 'x', label='Ref. TAC')
    ax[0].plot(np.mean(dt,axis=0), tac.tac, '.', label='Target TAC')
    ax[0].legend()
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Activity')


    ax[1].plot(tac.mft, tac.tac, 'b*', label='Target TAC')
    ax[1].plot(dt2mft(dt_gaps), tac.km_results['tacf'], 'r', label='Fit')
    ax[1].set_title(f'R1={r1:4.3e}; k2={k2:4.3e}; BP={bp:4.3e}')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Activity')

    ax[1].legend()

    plt.tight_layout()

    fpng = opth / f'{model}_fitting.png'
    plt.savefig(fpng, dpi=150, facecolor='auto', edgecolor='auto')

    return dict(r1=r1, k2=k2, bp=bp, TAC=tac, fig=fpng)



# =====================================================================
def km_img(
    fpet,
    rvoi,
    dt,
    dynamic_break=False,
    model='srtmb_basis',
    beta_lim=None,
    n_beta=100,
    k2p=None,
    thrshld=0.005,
    weights='dur',
    outpath=None):

    '''
    Run voxel-wise (image) kinetic modelling (KM) using NiftyPAD
    Arguments:
    fpet:       input NifTI file for dynamic PET image
    rvoi:       TAC of the reference region
    dt:         frame time definitions
    model:      model selection, one of the following: 
                'srtmb_basis', 'srtm', 'srtmb_k2p_basis', etc
    dynamic_break: if True, the so called coffee-break modelling is used
                to fill the gaps when measurement was not taken.
    beta_lim:   KM input parameter (see NiftyPAD).
    n_beta:     basis number (see NiftyPAD).
    k2p:        KM input parameter (see NiftyPAD)
    thrshld:    Threshold as percentage of the maximum PET value.
    weights:    KM weights for each dynamic frame; if set to 'dur'
                the duration of each frame will be used for setting up
                the weights; if set to None, no weights are used.

    '''

    #----------------------------------
    if not Path(fpet).is_file():
        raise IOError('the input PET file does not exist')
    else:
        petnii = nib.load(fpet)
        pet = petnii.get_fdata()
    #----------------------------------


    #---------------------------------- 
    if outpath is None:
        opth = fpet.parent
    else:
        opth = Path(outpath)
        nimpa.create_dir(opth)
    #----------------------------------


    # > reference region VOI, with interpolation
    ref = Ref(rvoi, dt)
    ref.interp_1cubic()

    #----- KM parameters -----
    if beta_lim is None:
        beta_lim = [0.01/60, 0.5/60]

    # >>> TODO: check this k2p what to use by default
    # if k2p is None:
    #     k2p = 0.00025

    if weights is None:
        w = None
    elif weights=='dur':
        w = dt2tdur(dt)
        w = w / np.amax(w)
    else:
        raise ValueError('currently unrecognised weights for dynamic frames')
    #-------------------------

    b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=w, k2p=k2p)
    user_inputs = {
        'dt':dt,
        'inputf1':ref.inputf1cubic,
        'w':None,
        'k2p':0.00025,
        'beta_lim':beta_lim,
        'n_beta':n_beta,
        'b': b}


    imr1 = np.zeros(petnii.shape[0:3])
    imk2 = np.zeros(petnii.shape[0:3])
    imbp = np.zeros(petnii.shape[0:3])

    # > TODO: get the threshold based on smoothed image
    thr = 0.005*np.amax(pet)

    model_inputs=get_model_inputs(user_inputs,model)
    km_outputs = ['R1', 'k2', 'BP']

    # > loop through dims to iteratively go over each voxel
    for zi in range(petnii.shape[2]):
        print("z-slice #:", str(zi))
        for yi in range(petnii.shape[1]):
            for xi in range(petnii.shape[0]):
                # > array of voxel TAC
                vtac = pet[xi,yi,zi,:]
                if np.mean(vtac) > thr:
                    voxtac = TAC(vtac, dt)
                    voxtac.run_model(model,model_inputs)

                    imr1[xi,yi,zi] = voxtac.km_results['r1']
                    imk2[xi,yi,zi] = voxtac.km_results['k2']
                    imbp[xi,yi,zi] = voxtac.km_results['bp']

    log.debug('Writing out KM parametric images')

    fr1 = opth/f'{model}-r1.nii.gz'
    fk2 = opth/f'{model}-k2.nii.gz'
    fbp = opth/f'{model}-bp.nii.gz'

    nib.save(nib.Nifti1Image(imr1, petnii.affine), fr1)
    nib.save(nib.Nifti1Image(imk2, petnii.affine), fk2)
    nib.save(nib.Nifti1Image(imbp, petnii.affine), fbp)


    return dict(fr1=fr1, fk2=fk2, fbp=fbp)
