'''
Aligning tools for PET dynamic frames
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging as log
import os
import shutil
from datetime import datetime
from pathlib import Path
from subprocess import run
import copy

from itertools import combinations
import dcm2niix
import numpy as np
import spm12
from niftypet import nimpa

from .utils import get_atlas
from .suvr_tools import preproc_suvr
from .preproc import id_acq
from .preproc import tracer_names

# > tracer in different radionuclide group
f18group = ['fbb', 'fbp', 'flute']
c11group = ['pib']

log.basicConfig(level=log.WARNING, format=nimpa.LOG_FORMAT)



# =====================================================================
def align_suvr(
    stat_tdata,
    outpath=None,
    not_aligned=True,
    reg_costfun='nmi',
    use_stored=False,
    reg_fwhm=8,
    reg_thrshld=2.0,
    com_correction=True
):
    '''
    Align SUVr frames after conversion to NIfTI format.

    Arguments:
    - reg_constfun: the cost function used in SPM registration/alignment of frames
    - reg_force:    force running the registration even if the registration results
                are already calculated and stored in the output folder.
    - reg_fwhm: the FWHM of the Gaussian kernel used for smoothing the images before
                registration and only for registration purposes.
    - reg_thrshld: the threshold for the registration metric (combined trans. and rots)
                when deciding to apply the transformation
    - com_correction: centre-of-mass correction - moves the coordinate system to the 
                centre of the spatial image intensity distribution.

    '''

    if outpath is None:
        align_out = stat_tdata[stat_tdata['descr']['frms'][0]]['fnii'].parent.parent
    else:
        align_out = Path(outpath)

    # > NIfTI output folder
    niidir = align_out / 'NIfTI_aligned'
    nimpa.create_dir(niidir)

    # > folder of resampled and aligned NIfTI files (SPM)
    rsmpl_opth = niidir / 'SPM-aligned'
    nimpa.create_dir(rsmpl_opth)

    tstudy = stat_tdata[stat_tdata['descr']['frms'][0]]['tstudy']

    # > re-aligned output file names and output dictionary
    faligned = 'SUVr_aligned_' + nimpa.rem_chars(stat_tdata[stat_tdata['descr']['frms'][0]]['series']) + '.nii.gz'
    faligned_c = 'SUVr_aligned_CoM-mod_' + nimpa.rem_chars(stat_tdata[stat_tdata['descr']['frms'][0]]['series']) + '.nii.gz'
    faligned_s = 'Aligned-Frames-to-SUVr_' + nimpa.rem_chars(stat_tdata[stat_tdata['descr']['frms'][0]]['series']) + '.nii.gz'
    falign_dct = f'Aligned-Frames-to-SUVr_study-{tstudy}_' + nimpa.rem_chars(stat_tdata[stat_tdata['descr']['frms'][0]]['series'])+'.npy'
    faligned = niidir / faligned
    faligned_s = niidir / faligned_s
    faligned_c = niidir / faligned_c
    falign_dct = niidir / falign_dct

    # > the same for the not aligned frames, if requested
    fnotaligned = 'SUVr_NOT_aligned_' + nimpa.rem_chars(stat_tdata[stat_tdata['descr']['frms'][0]]['series']) + '.nii.gz'
    fnotaligned = niidir / fnotaligned

    # > Matrices: motion metric + paths to affine
    R = S = None

    outdct = None

    # > check if the file exists
    if not use_stored or not falign_dct.is_file():

        # -----------------------------------------------
        # > list all NIfTI files
        nii_frms = []
        for k in stat_tdata['descr']['suvr']['frms']:
            nii_frms.append(stat_tdata[k]['fnii'])
        # -----------------------------------------------

        # -----------------------------------------------
        # > CORE ALIGNMENT OF SUVR FRAMES:

        # > frame-based motion metric (rotations+translation)
        R = np.zeros((len(nii_frms), len(nii_frms)), dtype=np.float32)

        # > paths to the affine files
        S = [[None for _ in range(len(nii_frms))] for _ in range(len(nii_frms))]

        # > go through all possible combinations of frame registration
        for c in combinations(stat_tdata['descr']['suvr']['frms'], 2):
            frm0 = stat_tdata['descr']['suvr']['frms'].index(c[0])
            frm1 = stat_tdata['descr']['suvr']['frms'].index(c[1])

            fnii0 = nii_frms[frm0]
            fnii1 = nii_frms[frm1]

            log.info(f'registration of frame #{frm0} and frame #{frm1}')

            # > one way registration
            spm_res = nimpa.coreg_spm(fnii0, fnii1, fwhm_ref=reg_fwhm, fwhm_flo=reg_fwhm,
                                      fwhm=[13, 13], costfun=reg_costfun,
                                      fcomment=f'_combi_{frm0}-{frm1}', outpath=niidir,
                                      visual=0, save_arr=False, del_uncmpr=True)

            S[frm0][frm1] = spm_res['faff']

            rot_ss = np.sum((180 * spm_res['rotations'] / np.pi)**2)**.5
            trn_ss = np.sum(spm_res['translations']**2)**.5
            R[frm0, frm1] = rot_ss + trn_ss

            # > the other way registration
            spm_res = nimpa.coreg_spm(fnii1, fnii0, fwhm_ref=reg_fwhm, fwhm_flo=reg_fwhm,
                                      fwhm=[13, 13], costfun=reg_costfun,
                                      fcomment=f'_combi_{frm1}-{frm0}', outpath=niidir,
                                      visual=0, save_arr=False, del_uncmpr=True)

            S[frm1][frm0] = spm_res['faff']

            rot_ss = np.sum((180 * spm_res['rotations'] / np.pi)**2)**.5
            trn_ss = np.sum(spm_res['translations']**2)**.5
            R[frm1, frm0] = rot_ss + trn_ss

        # > sum frames along floating frames
        fsum = np.sum(R, axis=0)

        # > sum frames along reference frames
        rsum = np.sum(R, axis=1)

        # > reference frame for SUVr composite frame
        rfrm = np.argmin(fsum + rsum)

        niiref = nimpa.getnii(nii_frms[rfrm], output='all')

        # > initialise target aligned SUVr image
        niiim = np.zeros((len(nii_frms),) + niiref['shape'], dtype=np.float32)

        # > copy in the target frame for SUVr composite
        niiim[rfrm, ...] = niiref['im']

        # > aligned individual frames, starting with the reference 
        fnii_aligned = [nii_frms[rfrm]]

        for ifrm in range(len(nii_frms)):
            if ifrm == rfrm:
                continue

            # > resample images for alignment
            frsmpl = nimpa.resample_spm(
                nii_frms[rfrm],
                nii_frms[ifrm],
                S[rfrm][ifrm],
                intrp=1.,
                outpath=rsmpl_opth,
                pickname='flo',
                del_ref_uncmpr=True,
                del_flo_uncmpr=True,
                del_out_uncmpr=True,
            )

            fnii_aligned.append(frsmpl)

            niiim[ifrm, ...] = nimpa.getnii(frsmpl)

        # > save aligned SUVr frames
        nimpa.array2nii(
            niiim, niiref['affine'], faligned, descrip='AmyPET: aligned SUVr frames',
            trnsp=(niiref['transpose'].index(0), niiref['transpose'].index(1),
                   niiref['transpose'].index(2)), flip=niiref['flip'])


        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        # SINGLE SUVR FRAME  &  CoM CORRECTION
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > preprocess the aligned PET into a single SUVr frame
        suvr_frm = preproc_suvr(faligned, outpath=niidir, com_correction=True)
        fref = suvr_frm['fcom']
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++

        # > saved frames aligned and CoM-modified
        fniic_aligned = []
        niiim[:] = 0
        for i in range(len(fnii_aligned)):
            com_ = nimpa.centre_mass_corr(fnii_aligned[i], outpath=niidir, com=suvr_frm['com'])
            fniic_aligned.append(com_['fim'])
            niiim[i, ...] = nimpa.getnii(com_['fim'])

        # > save aligned SUVr frames
        tmp = nimpa.getnii(fref, output='all')
        nimpa.array2nii(
            niiim, tmp['affine'], faligned_c, descrip='AmyPET: aligned SUVr frames, CoM-modified',
            trnsp=(tmp['transpose'].index(0), tmp['transpose'].index(1),
                   tmp['transpose'].index(2)), flip=tmp['flip'])
        # -----------------------------------------------

        # > output dictionary
        outdct = dict(suvr={}, static={})

        outdct['suvr'] = {
            'fpet': faligned_c,
            'fsuvr':fref,
            'fpeti':fniic_aligned,
            'outpath': niidir,
            'Metric': R,
            'faff': S}

        # > save static image which is not aligned
        if not_aligned:
            nii_noalign = np.zeros(niiim.shape, dtype=np.float32)
            for k, fnf in enumerate(nii_frms):
                nii_noalign[k, ...] = nimpa.getnii(fnf)

            nimpa.array2nii(
                nii_noalign, niiref['affine'], fnotaligned,
                descrip='AmyPET: unaligned SUVr frames',
                trnsp=(niiref['transpose'].index(0), niiref['transpose'].index(1),
                       niiref['transpose'].index(2)), flip=niiref['flip'])

            outdct['suvr']['fpet_notaligned'] = fnotaligned





        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        # The remaining frames of static or fully dynamic PET
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        # > number of static frames
        nsfrm = len(stat_tdata['descr']['frms'])

        # > output files for affines
        S_ = [None for _ in range(nsfrm)]

        # > motion metric for any remaining frames
        R_ = np.zeros(nsfrm)

        # > output paths of aligned images for the static part
        fnii_aligned_ = [None for _ in range(nsfrm)]

        niiim_ = np.zeros((nsfrm,) + niiref['shape'], dtype=np.float32)

        # > index/counter for SUVr frames
        fsi = 0

        for fi, frm in enumerate(stat_tdata['descr']['frms']):
            if not frm in stat_tdata['descr']['suvr']['frms']:

                fnii = stat_tdata[frm]['fnii']
                # > register frame to the reference
                spm_res = nimpa.coreg_spm(fref, fnii, fwhm_ref=reg_fwhm, fwhm_flo=reg_fwhm,
                                          fwhm=[13, 13], costfun=reg_costfun,
                                          fcomment=f'_ref-static', outpath=niidir,
                                          visual=0, save_arr=False, del_uncmpr=True, pickname='flo')

                S_[fi] = spm_res['faff']

                rot_ss = np.sum((180 * spm_res['rotations'] / np.pi)**2)**.5
                trn_ss = np.sum(spm_res['translations']**2)**.5
                R_[fi] = rot_ss + trn_ss

                # > align each frame through resampling 
                if R_[fi]>reg_thrshld:
                    # > resample images for alignment
                    fnii_aligned_[fi] = nimpa.resample_spm(
                        fref,
                        fnii,
                        S_[fi],
                        intrp=1.,
                        outpath=rsmpl_opth,
                        pickname='flo',
                        del_ref_uncmpr=True,
                        del_flo_uncmpr=True,
                        del_out_uncmpr=True,
                    )
                else:
                    fnii_aligned_[fi] = rsmpl_opth/fnii.name
                    shutil.copyfile(fnii, fnii_aligned_[fi])

                niiim_[fi, ...] = nimpa.getnii(fnii_aligned_[fi])
            
            else:
                # > already aligned as part of SUVr
                fnii_aligned_[fi] = Path(outdct['suvr']['fpeti'][fsi])
                S_[fi] = S[rfrm][fsi]
                R_[fi] = R[rfrm][fsi]
                niiim_[fi, ...] = niiim[fsi,...]
                fsi += 1

        # > save aligned static frames
        nimpa.array2nii(
            niiim_, niiref['affine'], faligned_s, descrip='AmyPET: aligned static frames',
            trnsp=(niiref['transpose'].index(0), niiref['transpose'].index(1),
                   niiref['transpose'].index(2)), flip=niiref['flip'])

        outdct['static'] = {
            'fpet': faligned_s,
            'fpeti':fnii_aligned_,
            'outpath': niidir,
            'Metric': R_,
            'faff': S_}

        np.save(falign_dct, outdct)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
    else:
        outdct = np.load(falign_dct, allow_pickle=True)
        outdct = outdct.item()
        # suvr_frm = preproc_suvr(faligned, outpath=niidir)
        # outdct = {}
        # outdct['suvr'] = dict(fpet=faligned, fsuvr=suvr_frm['fsuvr'], outpath=niidir)
        # outdct['static'] = dict(fpet=faligned_s, outpath=niidir)


    return outdct
# =====================================================================




# ========================================================================================
def align_break(
    niidat,
    aligned_suvr,
    frame_min_dur=60,
    reg_costfun='nmi',
    reg_fwhm=8,
    reg_thrshld=2.0,
    decay_corr=True,
    use_stored=False,
    ):
    
    ''' Align the Coffee-Break protocol data to Static/SUVr data
        to form one consistent dynamic 4D NIfTI image
        Arguments:
        - niidat:   dictionary of all input NIfTI series.
        - aligned_suvr: dictionary of the alignment output for SUVr frames
        - frame_min_dur: the shortest PET frame to be used for registration
                    in the alignment process.
        - reg_*:    SPM12 registration parameters.
        - reg_thrshld: the threshold of the metric of combined rotations
                    and translations to identify significant motion worth
                    correcting for.
        - decay_corr: correct for decay between different series relative
                    to the earliest one
    '''

    # > identify coffee-break data if any
    bdyn_tdata = id_acq(niidat, acq_type='break')

    if not bdyn_tdata:
        log.info('no coffee-break protocol data detected.')
        return aligned_suvr

    #-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
    if decay_corr:
        # > DECAY CORRECTION
        # > get the start time of each series for decay correction if requested
        ts = [sri['time'][0] for sri in niidat['descr']]
        # > index the earliest ref time
        i_tref = np.argmin(ts)
        idxs = list(range(len(ts)))
        idxs.pop(i_tref)
        i_tsrs = idxs
        # > time difference
        td = [ts[i]-ts[i_tref] for i in i_tsrs]
        if len(td)>1:
            raise ValueError('currently only one dynamic break is allowed - detected more than one')
        else:
            td = td[0]

        # > what tracer / radionuclide is used?
        istp = 'F18'*(niidat['tracer'] in f18group) + 'C11'*(niidat['tracer'] in c11group)

        # > decay constant using half-life
        lmbd = np.log(2) / nimpa.resources.riLUT[istp]['thalf']

        # > decay correction factor
        dcycrr = 1/np.exp(-lmbd * td)
    else:
        dcycrr = 1.
    #-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


    # > the shortest frames acceptable for registration
    frm_lsize = frame_min_dur

    # > output folder for mashed frames for registration/alignment
    mniidir = niidat['outpath']/'NIfTI_mashed'
    rsmpl_opth = mniidir/'SPM-aligned'
    nimpa.create_dir(mniidir)
    nimpa.create_dir(rsmpl_opth)


    tstudy = bdyn_tdata[bdyn_tdata['descr']['frms'][0]]['tstudy']
    
    # > output dictionary and NIfTI files
    falign_dct = niidat['outpath']/'NIfTI_aligned'/f'Dynamic-early-frames_study-{tstudy}_aligned-to-SUVr-ref.npy'
    falign_nii = niidat['outpath']/'NIfTI_aligned'/f'Dynamic-early-frames_study-{tstudy}_aligned-to-SUVr-ref.nii.gz'
    
    if use_stored and falign_dct.is_file():
        outdct = np.load(falign_dct, allow_pickle=True)
        return outdct.item()


    # > get the aligned static NIfTI files
    faligned_stat = aligned_suvr['static']['fpeti']

    # > reference frame (SUVr by default)
    fref = aligned_suvr['suvr']['fsuvr']

    # > dictionary of coffee-break dynamic frames
    snii_dct = {k:bdyn_tdata[k]['fnii'] for k in bdyn_tdata['descr']['frms']}

    # > number of frames
    nfrm = len(snii_dct)

    # > timings of the frames
    ts = np.array(bdyn_tdata['descr']['timings'])

    # > each frame duration
    dur = np.zeros(nfrm)
    dur = np.array([t[1]-t[0] for t in ts])

    # > lowest frame size for registration (in seconds)
    frms_l = dur<frm_lsize

    # > number of frames to be mashed for registration
    nmfrm = np.sum(frms_l)

    # > number of resulting mashed frame sets
    nfrm_l = int(np.floor(np.sum(dur[frms_l])/frm_lsize))

    # > overall list of mashed frames and normal frames
    #   which are longer than `frm_lsize`
    mfrms = []

    nmfrm_chck = 0
    # > mashing frames for registration
    for i in range(nfrm_l):
        sfrms = ts[:,1]<=(i+1)*frm_lsize
        sfrms *= ts[:,1]> i*frm_lsize

        # > list of keys of frames to be mashed
        k_mfrm = [k for i,k in enumerate(bdyn_tdata['descr']['frms']) if sfrms[i]]

        # > append to the overall list of mashed frames
        mfrms.append(k_mfrm)

        # > update the overall number of frames to be mashed
        nmfrm_chck += len(k_mfrm)

    #---------------------
    if nmfrm_chck!=nmfrm:
        raise ValueError('Mashing frames inconsistent: number of frames to be mashed incorrectly established.')
    #---------------------
    
    # > add the normal length (not mashed) frames
    for i,frm in enumerate(~frms_l):
        if frm:
            mfrms.append([bdyn_tdata['descr']['frms'][i]])
            nmfrm_chck += 1

    #---------------------
    if nmfrm_chck!=len(bdyn_tdata['descr']['frms']):
        raise ValueError('Mashing frames inconsistency: number of overall frames, including mashed, is incorrectly established.')
    #---------------------
    

    # > the output file paths of mashed frames
    mfrms_out = []

    # > generate NIfTI series of mashed frames for registration
    for mgrp in mfrms:

        # > image holder
        tmp = nimpa.getnii(snii_dct[mgrp[0]], output='all')
        im = np.zeros(tmp['shape'], dtype=np.float32)
        
        for frm in mgrp:
            im += nimpa.getnii(snii_dct[frm])

        # > output file path and name
        fout = mniidir/(f'mashed_n{len(mgrp)}_' + snii_dct[mgrp[0]].name)

        # > append the mashed frames output file path
        mfrms_out.append(fout)

        nimpa.array2nii(
            im,
            tmp['affine'],
            fout,
            descrip='mashed PET frames for registration',
            trnsp=tmp['transpose'],
            flip=tmp['flip'])

    if len(mfrms_out)!=len(mfrms):
        raise ValueError('The number of generated mashed frames is inconsistent with the intended mashed frames')


    # > initialise the array for metric of registration result (sum of angles+translations)
    R = np.zeros(len(mfrms_out))

    # > affine file outputs
    S = [None for _ in range(len(mfrms_out))]

    # > aligned/resampled file names
    faligned = [None for _ in range(nfrm)]

    # > counter for base frames
    fi = 0

    # > register the mashed frames to the reference (SUVr frame by default)
    for mi, mfrm in enumerate(mfrms_out):

        # > make sure the images are not compressed, i.e., ending with .nii
        if not mfrm.name.endswith('.nii'):
            raise ValueError('The mashed frame files should be uncompressed NIfTI')

        # > register mashed frames to the reference
        spm_res = nimpa.coreg_spm(fref, mfrm, fwhm_ref=reg_fwhm, fwhm_flo=reg_fwhm,
                                  fwhm=[13, 13], costfun=reg_costfun,
                                  fcomment=f'_mashed_ref-mfrm', outpath=mniidir,
                                  visual=0, save_arr=False, del_uncmpr=True, pickname='flo')

        S[mi] = (spm_res['faff'])

        rot_ss = np.sum((180 * spm_res['rotations'] / np.pi)**2)**.5
        trn_ss = np.sum(spm_res['translations']**2)**.5
        R[mi] = rot_ss + trn_ss

        # > align each frame through resampling 
        for frm in mfrms[mi]:
            if R[mi]>reg_thrshld:
                # > resample images for alignment
                faligned[fi] = nimpa.resample_spm(
                    fref,
                    snii_dct[frm],
                    S[mi],
                    intrp=1.,
                    outpath=rsmpl_opth,
                    pickname='flo',
                    del_ref_uncmpr=True,
                    del_flo_uncmpr=True,
                    del_out_uncmpr=True,
                )

            else:
                faligned[fi] = rsmpl_opth/frm.name
                shutil.copyfile(frm, faligned[fi])

            fi += 1

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > number of frames for the whole study
    nfrma = len(faligned+faligned_stat)
    tmp = nimpa.getnii(faligned[0], output='all')
    niia = np.zeros((nfrma,)+tmp['shape'], dtype=np.float32)
    for fi, frm in enumerate(faligned):
        im_ = nimpa.getnii(frm)
        # > remove NaNs if any
        im_[np.isnan(im_)] = 0
        niia[fi, ...] = im_

    for fii, frm in enumerate(faligned_stat):
        im_ = dcycrr * nimpa.getnii(frm)
        # > remove NaNs if any
        im_[np.isnan(im_)] = 0
        niia[fi+fii, ...] = im_


    # > save aligned SUVr frames
    nimpa.array2nii(
        niia,
        tmp['affine'],
        falign_nii,
        descrip='AmyPET: aligned dynamic frames',
        trnsp=(tmp['transpose'].index(0), tmp['transpose'].index(1),
               tmp['transpose'].index(2)),
        flip=tmp['flip'])

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    outdct = dict(fpet=falign_nii, fpeti=faligned+faligned_stat, mashed_frms=mfrms)
    np.save(falign_dct, outdct)

    return outdct
# ========================================================================================

