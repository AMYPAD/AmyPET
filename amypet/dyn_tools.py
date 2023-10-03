'''
Static frames processing tools for AmyPET
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.optimize import fmin
from niftypet import nimpa
from tqdm import trange

from niftypad import basis
from niftypad.kt import dt2mft, dt2tdur, dt_fill_gaps
from niftypad.models import get_model_inputs
from niftypad.tac import TAC, Ref

log = logging.getLogger(__name__)


def timing_dyn(niidat):
    '''
    Get the frame timings from the NIfTI series
    '''

    t = []
    for dct in niidat['descr']:
        t += dct['timings']
    t.sort()
    # > time mid points
    tm = [(tt[0] + tt[1]) / 2 for tt in t]

    # > convert the timings to NiftyPAD format
    dt = np.array(t).T

    return {'timings': t, 'niftypad': dt, 't_mid_points': tm}



#==========================================================
def dyn_timing(dat):
    ''' establish the timing per frame in minutes
    '''

    if 'descr' in dat:
        t = []
        for dct in dat['descr']:
            t += dct['timings']
        t.sort()
        timings = np.array(t)
    elif isinstance(dat, (list, np.ndarray)):
        timings = np.array(dat)
    else:
        raise ValueError('wrong format for the time data')


    # > time point for each frame
    tp = np.mean(timings, axis=1) / 60

    # > frame duration
    dtp = np.array([t[1]-t[0] for t in timings])/60

    # > number of frame/time points
    nt = len(tp)

    return dict(tp=tp, dtp=dtp, nt=nt)
#==========================================================


#==========================================================
def fit_tac(tac, tp, plotting=True):
    ''' Fit exponential to TAC data points
    '''

    # > get starting parameters
    i1 = np.argmax(tac)
    a1 = np.max(tac)
    t1 = tp[i1]
    t0 = tp[np.nonzero(tac<(a1/100))]
    t0 = t0[-1]
    pp0 = [t0, t1, a1-1, 0.01, 1]

    def obj_fun(pp):
        ''' objective function
        '''
        ff = test_fun(pp)
        return np.sum((ff-tac)**2)

    def test_fun(pp):
        ''' test function
        '''
        t0 = pp[0]
        t1 = pp[1]
        npp = len(pp)
        ne = int(np.ceil( (npp-2)/2 ))
        aa = np.zeros(ne)
        ss = np.zeros(ne)

        for j in range(ne):
            ip1 = 2*j + 2
            ip2 = ip1 + 1
            aa[j] = pp[ip1]
            if ip2 < npp:
                ss[j] = pp[ip2]

        ff = np.zeros(len(tp))
        ii = (tp>=t0) & (tp<=t1)
        ff[ii] = (tp[ii]-t0) * np.sum(aa)/(t1-t0)
        ii = tp>t1
        for j in range(ne):    
            ff[ii] = ff[ii] + aa[j] * np.exp(-ss[j]*(tp[ii]-t1))

        return ff

    pp_opt = fmin(obj_fun, pp0)
    yy_opt = test_fun(pp_opt)

    if plotting:
        fig, ax = plt.subplots(1,1, figsize=(9,6))
        ax.plot(tp, tac, '+')
        ax.plot(tp, yy_opt)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Activity conc.')
        ax.set_title('Reference region')

    return dict(y=yy_opt, pars=pp_opt)   
#==========================================================

#==========================================================
def logan_fit(ref_tac, tac, t, t_star, plot=True):
    ''' fit logan model to dynamic PET data
    '''

    if t['nt']!=len(ref_tac) or t['nt']!=len(tac):
        raise ValueError('incompatible TAC and/or reference TAC and time vector lengths')

    S_ref = np.cumsum(t['dtp']*ref_tac)
    S_tac = np.cumsum(t['dtp']*tac)

    aa = np.zeros(t['nt'])
    bb = np.zeros(t['nt'])
    
    ii = tac>0
    aa[ii] = S_ref[ii]/tac[ii]
    bb[ii] = S_tac[ii]/tac[ii]

    ii = ( t['tp'] > t_star )
    pp = np.polyfit(aa[ii], bb[ii], 1)

    xx = np.array([0, np.max(aa)])

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(9,6))
        ax.plot(aa, bb,'+', label='time points')
        ax.plot(xx, pp[0]*xx+pp[1], label='fit')
        ax.set_xlabel('Logan-x (min)')
        ax.set_ylabel('Logan-y (min)')
        ax.set_title('TAC')

    return dict(pars=pp, aa=aa, bb=bb, t_star=t_star)
#==========================================================





# =====================================================================
def km_voi(dynvois, voi=None, dynamic_break=False, model='srtmb_basis', beta_lim=None, n_beta=100,
           k2p=None, weights='dur', outpath=None):
    """
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
    """

    if outpath is None:
        opth = dynvois['outpath']
    else:
        opth = Path(outpath)
        nimpa.create_dir(opth)

    # > start/stop times for each frame
    dt = dynvois['dt']

    # > reference region VOI, with interpolation
    ref = Ref(dynvois['voi']['cerebellum']['avg'], dt)
    ref.interp_1cubic()

    # > target TAC of selected VOI
    tac = TAC(dynvois['voi'][voi]['avg'], dt)

    # ----- KM parameters -----
    if beta_lim is None:
        beta_lim = [0.01 / 60, 0.5 / 60]

    if weights is None:
        w = None
    elif weights == 'dur':
        w = dt2tdur(dt)
        w = w / np.amax(w)
    else:
        raise ValueError('currently unrecognised weights for dynamic frames')
    # -------------------------

    b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=w, k2p=k2p)
    user_inputs = {
        'dt': dt, 'inputf1': ref.inputf1cubic, 'w': w, 'k2p': k2p, 'beta_lim': beta_lim,
        'n_beta': n_beta, 'b': b}

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
    ax[0].plot(np.mean(dt, axis=0), ref.tac, 'x', label='Ref. TAC')
    ax[0].plot(np.mean(dt, axis=0), tac.tac, '.', label='Target TAC')
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

    return {'r1': r1, 'k2': k2, 'bp': bp, 'TAC': tac, 'fig': fpng}


# =====================================================================
def km_img(fpet, rvoi, dt, dynamic_break=False, model='srtmb_basis', beta_lim=None, n_beta=100,
           k2p=None, thrshld=0.005, weights='dur', outpath=None):
    """
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

    """
    if not Path(fpet).is_file():
        raise IOError('the input PET file does not exist')
    else:
        petnii = nib.load(fpet)
        pet = petnii.get_fdata()

    if outpath is None:
        opth = fpet.parent
    else:
        opth = Path(outpath)
        nimpa.create_dir(opth)

    # > reference region VOI, with interpolation
    ref = Ref(rvoi, dt)
    ref.interp_1cubic()

    # ----- KM parameters -----
    if beta_lim is None:
        beta_lim = [0.01 / 60, 0.5 / 60]

    # >>> TODO: check this k2p what to use by default
    # if k2p is None:
    #     k2p = 0.00025

    if weights is None:
        w = None
    elif weights == 'dur':
        w = dt2tdur(dt)
        w = w / np.amax(w)
    else:
        raise ValueError('currently unrecognised weights for dynamic frames')
    # -------------------------

    b = basis.make_basis(ref.inputf1cubic, dt, beta_lim=beta_lim, n_beta=n_beta, w=w, k2p=k2p)
    user_inputs = {
        'dt': dt, 'inputf1': ref.inputf1cubic, 'w': None, 'k2p': k2p, 'beta_lim': beta_lim,
        'n_beta': n_beta, 'b': b}

    imr1 = np.zeros(petnii.shape[0:3])
    imk2 = np.zeros(petnii.shape[0:3])
    imbp = np.zeros(petnii.shape[0:3])

    # > TODO: get the threshold based on smoothed image
    thr = 0.005 * np.amax(pet)

    model_inputs = get_model_inputs(user_inputs, model)

    # > loop through dims to iteratively go over each voxel
    for zi in trange(petnii.shape[2], desc='image z-slice'):
        # print("z-slice #:", str(zi))
        for yi in range(petnii.shape[1]):
            for xi in range(petnii.shape[0]):
                # > array of voxel TAC
                vtac = pet[xi, yi, zi, :]
                if np.mean(vtac) > thr:
                    voxtac = TAC(vtac, dt)
                    voxtac.run_model(model, model_inputs)

                    imr1[xi, yi, zi] = voxtac.km_results['r1']
                    imk2[xi, yi, zi] = voxtac.km_results['k2']
                    imbp[xi, yi, zi] = voxtac.km_results['bp']

    log.debug('Writing out KM parametric images')

    fr1 = opth / f'{model}-r1.nii.gz'
    fk2 = opth / f'{model}-k2.nii.gz'
    fbp = opth / f'{model}-bp.nii.gz'

    nib.save(nib.Nifti1Image(imr1, petnii.affine), fr1)
    nib.save(nib.Nifti1Image(imk2, petnii.affine), fk2)
    nib.save(nib.Nifti1Image(imbp, petnii.affine), fbp)

    return {'fr1': fr1, 'fk2': fk2, 'fbp': fbp}
