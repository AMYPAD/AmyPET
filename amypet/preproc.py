'''
Preprocessing tools for AmyPET core processes
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022"

import logging as log
import os
import shutil
from datetime import datetime
from itertools import combinations
from pathlib import Path
from subprocess import run

import dcm2niix
import numpy as np
import spm12
from niftypet import nimpa

from .suvr_tools import r_trimup
from .utils import get_atlas

log.basicConfig(level=log.WARNING, format=nimpa.LOG_FORMAT)

# ------------------------------------------------
# DEFINITIONS:
# TODO: move these to a separate file, e.g., `defs.py`

# > SUVr time window post injection and duration
suvr_twindow = {
                                                       # yapf: ignore
    'pib': [90 * 60, 110 * 60, 1200],
    'flute': [90 * 60, 110 * 60, 1200],
    'fbb': [90 * 60, 110 * 60, 1200],
    'fbp': [50 * 60, 60 * 60, 600]}
tracer_names = {
                                                       # yapf: ignore
    'pib': ['pib'],
    'flute': ['flt', 'flut', 'flute', 'flutemetamol'],
    'fbb': ['fbb', 'florbetaben'],
    'fbp': ['fbp', 'florbetapir']}

# > break time for coffee break protocol (target)
break_time = 1800

# > time margin for the 1st coffee break acquisition
breakdyn_t = (1200, 2400)

# > minimum time for the full dynamic acquisition
fulldyn_time = 3600

# ------------------------------------------------


# =====================================================================
def explore_input(input_fldr, tracer=None, suvr_win_def=None, outpath=None, margin=0.1):
    '''
    Process the input folder of amyloid PET DICOM data.
    The folder can contain two subfolders for a coffee break protocol including
    early dynamic followed by a static scan.
    The folder can also contain static or dynamic DICOM files.
    Those files can also be within a subfolder.

    Return the dictionary of (1) the list of dictionaries for each DICOM folder
    (2) list of descriptions for each DICOM folder for classification of input

    Arguments:
    - tracer:   The name of one of the three tracers: 'flute', 'fbb', 'fbp'
    - suvr_win_def: The definition of SUVr time frame (SUVr/CL is always calculated)
                as a two-element list [t_start, t_stop] in seconds.  If the
                window is not defined the function will attempt to get the
                information from the tracer info and use the default (as
                defined in`defs.py`)
    - outpath:  output path where all the intermediate and final results are
                stored.
    - margin:   margin used for accepting SUVr time windows (0.1 corresponds to 10%)
    '''

    # > make the input a Path object
    input_fldr = Path(input_fldr)

    if not input_fldr.is_dir():
        raise ValueError('Incorrect input - not a folder!')

    if outpath is None:
        amyout = input_fldr.parent / 'amypet_output'
    else:
        amyout = Path(outpath)
    nimpa.create_dir(amyout)

    # ================================================
    # > first check if the folder has DICOM series

    # > multiple series in folders (if any)
    msrs = []
    for itm in input_fldr.iterdir():
        if itm.is_dir():
            srs = nimpa.dcmsort(itm, grouping='a+t+d', copy_series=True, outpath=amyout)
            if srs:
                msrs.append(srs)

    # > check files in the input folder
    srs = nimpa.dcmsort(input_fldr, grouping='a+t+d', copy_series=True, outpath=amyout)
    if srs:
        msrs.append(srs)
    # ================================================

    # > initialise the list of acquisition classification
    msrs_class = []

    # > time-sorted series
    msrs_t = []

    for m in msrs:

        # > for each folder do the following:

        # > time sorted series according to acquisition time
        srs_t = {k: v for k, v in sorted(m.items(), key=lambda item: item[1]['tacq'])}

        msrs_t.append(srs_t)

        # -----------------------------------------------
        # > frame timings relative to the injection time -
        #   radiopharmaceutical administration start time
        t_frms = []
        for k in srs_t:
            t0 = datetime.strptime(srs_t[k]['dstudy'] + srs_t[k]['tacq'],
                                   '%Y%m%d%H%M%S') - srs_t[k]['radio_start_time']
            t1 = datetime.strptime(
                srs_t[k]['dstudy'] + srs_t[k]['tacq'],
                '%Y%m%d%H%M%S') + srs_t[k]['frm_dur'] - srs_t[k]['radio_start_time']
            t_frms.append((t0.seconds, t1.seconds))

        t_starts = [t[0] for t in t_frms]
        t_stops = [t[1] for t in t_frms]

        # > overall acquisition duration
        acq_dur = t_frms[-1][-1] - t_frms[0][0]
        # -----------------------------------------------

        # # -----------------------------------------------
        # # > image path (input)
        # impath = srs_t[next(iter(srs_t))]['files'][0].parent
        # # -----------------------------------------------

        # > check if the frames qualify for static, fully dynamic or
        # > coffee-break dynamic acquisition
        acq_type = None
        if t_frms[0][0] < 1:
            if t_frms[-1][-1] > breakdyn_t[0] and t_frms[-1][-1] <= breakdyn_t[1]:
                acq_type = 'breakdyn'
            elif t_frms[-1][-1] >= fulldyn_time:
                acq_type = 'fulldyn'
        elif t_frms[0][0] > 1:
            acq_type = 'static'
        # -----------------------------------------------

        # > classify tracer if possible and if not given
        if tracer is None:
            if 'tracer' in srs_t[next(iter(srs_t))]:
                tracer_dcm = srs_t[next(iter(srs_t))]['tracer'].lower()
                for t in tracer_names:
                    for n in tracer_names[t]:
                        if n in tracer_dcm:
                            tracer = t

            # > when tracer info not provided and not in DICOMs
            if acq_type == 'static' and not tracer:

                # > assuming the first of the following tracers then
                for t in suvr_twindow:
                    dur = suvr_twindow[t][2]
                    if (acq_dur > dur * (1-margin)) and (t_frms[0][0] < suvr_twindow[t][0] *
                                                         (1+margin)):
                        tracer = t
                        break
        else:
            tracer_grp = [tracer in tracer_names[t] for t in tracer_names]
            if any(tracer_grp):
                tracer = np.array(list(tracer_names.keys()))[tracer_grp][0]
        # -----------------------------------------------

        # > is the static acquisition covering the provided SUVr frame definition?
        if acq_type == 'static':
            # -----------------------------------------------
            # > try to establish the SUVr window even if not provided
            if suvr_win_def is None and not tracer:
                raise ValueError(
                    'Impossible to figure out tracer and SUVr time window - please specify them!')
            elif suvr_win_def is None and tracer:
                suvr_win = suvr_twindow[tracer][:2]
            else:
                suvr_win = suvr_win_def
            # -----------------------------------------------

            # > SUVr window margins, relative to the frame start time and the duration
            mrgn_suvr_start = margin * suvr_twindow[tracer][0]
            mrgn_suvr_dur = margin * suvr_twindow[tracer][2]

            if t_frms[0][0] < suvr_win[0] + mrgn_suvr_start and \
                t_frms[0][0] > suvr_win[0] - mrgn_suvr_start and \
                acq_dur > suvr_twindow[tracer][2] - mrgn_suvr_dur:

                t0_suvr = min(t_starts, key=lambda x: abs(x - suvr_win[0]))
                t1_suvr = min(t_stops, key=lambda x: abs(x - suvr_win[1]))

                frm_0 = t_starts.index(t0_suvr)
                frm_1 = t_stops.index(t1_suvr)

                msrs_class.append({
                    'acq': [acq_type, 'suvr'], 'time': (t0_suvr, t1_suvr), 'timings': t_frms,
                    'idxs': (frm_0, frm_1),
                    'frms': [s for i, s in enumerate(srs_t) if i in range(frm_0, frm_1 + 1)]})
            else:
                log.warning('The acquisition does not cover the requested time frame!')

                msrs_class.append({
                                                        #'inpath':impath,
                    'acq': [acq_type],
                    'time': (t_starts[0], t_stops[-1]),
                    'idxs': (0, len(t_frms) - 1),
                    'frms': [s for i, s in enumerate(srs_t)]})
        elif acq_type == 'breakdyn':
            t0_dyn = min(t_starts, key=lambda x: abs(x - 0))
            t1_dyn = min(t_stops, key=lambda x: abs(x - break_time))

            frm_0 = t_starts.index(t0_dyn)
            frm_1 = t_stops.index(t1_dyn)

            msrs_class.append({
                                          #'inpath':impath,
                'acq': [acq_type],
                'time': (t0_dyn, t1_dyn),
                'timings': t_frms,
                'idxs': (frm_0, frm_1),
                'frms': [s for i, s in enumerate(srs_t) if i in range(frm_0, frm_1 + 1)]})
        elif acq_type == 'fulldyn':
            t0_dyn = min(t_starts, key=lambda x: abs(x - 0))
            t1_dyn = min(t_stops, key=lambda x: abs(x - fulldyn_time))

            frm_0 = t_starts.index(t0_dyn)
            frm_1 = t_stops.index(t1_dyn)

            msrs_class.append({
                                          #'inpath':impath,
                'acq': [acq_type],
                'time': (t0_dyn, t1_dyn),
                'timings': t_frms,
                'idxs': (frm_0, frm_1),
                'frms': [s for i, s in enumerate(srs_t) if i in range(frm_0, frm_1 + 1)]})

    return {'series': msrs_t, 'descr': msrs_class, 'outpath': amyout, 'tracer': tracer}


# =====================================================================
def align_suvr(
    suvr_tdata,
    suvr_descr,
    outpath=None,
    not_aligned=True,
    reg_costfun='nmi',
    reg_force=False,
    reg_fwhm=8,
):
    '''
    Align SUVr frames after conversion to NIfTI format.

    Arguments:
    - reg_constfun: the cost function used in SPM registration/alignment of frames
    - reg_force:    force running the registration even if the registration results
                are already calculated and stored in the output folder.
    - reg_fwhm: the FWHM of the Gaussian kernel used for smoothing the images before
                registration and only for registration purposes.

    '''

    if outpath is None:
        align_out = suvr_tdata[next(iter(suvr_tdata))]['files'][0].parent.parent
    else:
        align_out = Path(outpath)

    # > NIfTI output folder
    niidir = align_out / 'NIfTI_SUVr'
    nimpa.create_dir(niidir)

    # > folder of resampled and aligned NIfTI files (SPM)
    rsmpl_opth = niidir / 'SPM-aligned'
    nimpa.create_dir(rsmpl_opth)

    # > the name of the output re-aligned file name
    faligned = 'SUVr_aligned_' + nimpa.rem_chars(suvr_tdata[next(
        iter(suvr_tdata))]['series']) + '.nii.gz'
    faligned = niidir / faligned

    # > the same for the not aligned frames, if requested
    fnotaligned = 'SUVr_NOT_aligned_' + nimpa.rem_chars(suvr_tdata[next(
        iter(suvr_tdata))]['series']) + '.nii.gz'
    fnotaligned = niidir / fnotaligned

    # > Matrices: motion metric + paths to affine
    R = S = None

    outdct = None

    # > check if the file exists
    if reg_force or not faligned.is_file():

        # > remove any files from previous runs
        files = niidir.glob('*')
        for f in files:
            if f.is_file():
                os.remove(f)
            else:
                shutil.rmtree(f)

        # > output nifty frame files
        nii_frms = []

        # -----------------------------------------------
        # > convert the individual DICOM frames to NIfTI
        for i, k in enumerate(suvr_descr['frms']):

            run([
                dcm2niix.bin, '-i', 'y', '-v', 'n', '-o', niidir, 'f', '%f_%s',
                suvr_tdata[k]['files'][0].parent])

            # > get the converted NIfTI file
            fnii = list(niidir.glob(str(suvr_tdata[k]['tacq']) + '*.nii*'))
            if len(fnii) != 1:
                raise ValueError('Unexpected number of converted NIfTI files')
            else:
                nii_frms.append(fnii[0])
        # -----------------------------------------------

        # -----------------------------------------------
        # > CORE ALIGNMENT OF SUVR FRAMES:

        # > frame-based motion metric (rotations+translation)
        R = np.zeros((len(nii_frms), len(nii_frms)), dtype=np.float32)

        # > paths to the affine files
        S = [[None for _ in range(len(nii_frms))] for _ in range(len(nii_frms))]

        # > go through all possible combinations of frame registration
        for c in combinations(suvr_descr['frms'], 2):
            frm0 = suvr_descr['frms'].index(c[0])
            frm1 = suvr_descr['frms'].index(c[1])

            fnii0 = nii_frms[frm0]
            fnii1 = nii_frms[frm1]

            log.info(f'registration of frame #{frm0} and frame #{frm1}')

            # > one way registration
            spm_res = nimpa.coreg_spm(fnii0, fnii1, fwhm_ref=reg_fwhm, fwhm_flo=reg_fwhm,
                                      fwhm=[13, 13], costfun=reg_costfun,
                                      fcomment=f'_combi_{frm0}-{frm1}', outpath=fnii0.parent,
                                      visual=0, save_arr=False, del_uncmpr=True)

            S[frm0][frm1] = spm_res['faff']

            rot_ss = np.sum((180 * spm_res['rotations'] / np.pi)**2)**.5
            trn_ss = np.sum(spm_res['translations']**2)**.5
            R[frm0, frm1] = rot_ss + trn_ss

            # > the other way registration
            spm_res = nimpa.coreg_spm(fnii1, fnii0, fwhm_ref=reg_fwhm, fwhm_flo=reg_fwhm,
                                      fwhm=[13, 13], costfun=reg_costfun,
                                      fcomment=f'_combi_{frm1}-{frm0}', outpath=fnii0.parent,
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

            niiim[ifrm, ...] = nimpa.getnii(frsmpl)

        # > save aligned SUVr frames
        nimpa.array2nii(
            niiim, niiref['affine'], faligned, descrip='AmyPET: aligned SUVr frames',
            trnsp=(niiref['transpose'].index(0), niiref['transpose'].index(1),
                   niiref['transpose'].index(2)), flip=niiref['flip'])
        # -----------------------------------------------

        outdct = {'fpet': faligned, 'outpath': niidir, 'Metric': R, 'faff': S}

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

            outdct['fpet_notaligned'] = fnotaligned

    return outdct


# =====================================================================


# =====================================================================
def native_proc(cl_dct, atlas='aal', res='1', outpath=None, refvoi_idx=None, refvoi_name=None):
    '''
    Preprocess SPM GM segmentation (from CL output) and AAL atlas
    to native PET space which is trimmed and upscaled to MR resolution.

    cl_dct:     CL process output dictionary as using the normalisation
                and segmentation from SPM as used in CL.
    refvoi_idx: indices of the reference region specific for the chosen
                atlas.  The reference VOI will then be separately
                provided in the output.
    refvoi_name: name of the reference region/VOI

    '''

    # > output path
    if outpath is None:
        natout = cl_dct['opth'].parent.parent
    else:
        natout = Path(outpath)

    # > get the AAL atlas with the resolution of 1 mm
    fatl = get_atlas(atlas='aal', res=1, outpath=natout)

    # > trim and upscale the native PET relative to MR resolution
    trmout = r_trimup(cl_dct['petc']['fim'], cl_dct['mric']['fim'], outpath=natout,
                      store_img_intrmd=True)

    # > get the trimmed PET as dictionary
    petdct = nimpa.getnii(trmout['ftrm'], output='all')
    # > SPM bounding box of the PET image
    ml = spm12.get_matlab()
    bbox = spm12.get_bbox(petdct)

    # > get the inverse affine transform to PET native space
    M = np.linalg.inv(cl_dct['reg2']['affine'])
    Mm = ml.double(M.tolist())

    # > copy the inverse definitions to be modified with affine to native PET space
    fmod = shutil.copyfile(cl_dct['norm']['invdef'],
                           cl_dct['norm']['invdef'].split('.')[0] + '_2nat.nii')
    eng = spm12.ensure_spm('')
    eng.amypad_coreg_modify_affine(fmod, Mm)

    # > unzip the atlas and transform it to PET space
    fniiatl = nimpa.nii_ugzip(fatl, outpath=natout)

    # > inverse transform the atlas to PET space
    finvatl = spm12.normw_spm(fmod, [fniiatl + ',1'], voxsz=1., intrp=0., bbox=bbox,
                              outpath=natout)

    # > remove the uncompressed input atlas after transforming it
    os.remove(fniiatl)

    # > GM mask
    fgmpet = spm12.resample_spm(trmout['ftrm'], cl_dct['norm']['c1'], M, intrp=1.0, outpath=natout,
                                pickname='flo', fcomment='_GM_in_PET', del_ref_uncmpr=True,
                                del_flo_uncmpr=True, del_out_uncmpr=True)

    gm_msk = nimpa.getnii(fgmpet)
    atl_im = nimpa.getnii(finvatl[0])

    # > get a probability mask for cerebellar GM

    if refvoi_idx is not None:
        refmsk = np.zeros(petdct['im'].shape, dtype=np.float32)
        for mi in refvoi_idx:
            print(mi)
            msk = atl_im == mi
            refmsk += msk * gm_msk

    # > probability mask for chosen VOI
    if refvoi_name is not None:
        fpmsk = natout / (refvoi_name+'_probmask.nii.gz')
    else:
        fpmsk = natout / 'reference_VOI_probmask.nii.gz'

    # > save the mask to NIfTI file
    nimpa.array2nii(
        refmsk, petdct['affine'], fpmsk, descrip='AmyPET: probability mask',
        trnsp=(petdct['transpose'].index(0), petdct['transpose'].index(1),
               petdct['transpose'].index(2)), flip=petdct['flip'])

    return {
        'fpet': trmout['ftrm'], 'outpath': natout, 'finvdef': fmod, 'fatl': finvatl, 'fgm': fgmpet,
        'atlas': atl_im, 'gm_msk': gm_msk, 'frefvoi': fpmsk, 'refvoi': refmsk}



# =====================================================================
# > PREPARE FOR VISUAL READING
def vr_proc(
        fpet,
        fmri,
        pet_affine=np.eye(4),
        mri_affine=np.eye(4),
        intrp= 1.0,
        ref_voxsize = 1.0,
        ref_imsize=256,
        fref=None,
        outfref=None,
        outpath=None,
        fcomment=''):
    '''
    Generate PET and the accompanying MRI images for amyloid visual reads aligned (rigidly) 
    to the MNI space.

    Arguments:
    - fpet:     the PET file name
    - fmri:     the MRI (T1w) file name
    - pet_affine:   PET affine given as a file path or a Numpy array 
    - mri_affine:   MRI affine given as a file path or a Numpy array 
    - intrp:    interpolation level used in resampling (0:NN, 1: trilinear, etc.) 
    - ref_voxsize:  the reference voxel size isotropically (default 1.0 mm)
    - ref_imsize:   the reference image size isotropically (default (256))
    - fref:     the reference image file path instead of the two above
    - outpath:  the output path
    - fcomment: the output file name suffix/comment 
    - outfref:  if reference given using `ref_voxsize` and `ref_imsize` instead of 
                reference file path `ferf`, the reference image will be save to
                this path.
    '''


    if os.path.isfile(fpet) and os.path.isfile(fmri):
        fpet = Path(fpet)
        fmri = Path(fmri)
    else:
        raise ValueError('Incorrect PET and/or MRI file paths!')


    if not isinstance(pet_affine, np.ndarray) and not isinstance(pet_affine, (str, pathlib.PurePath)):
        raise ValueError('Incorrect PET affine input')

    if not isinstance(mri_affine, np.ndarray) and not isinstance(mri_affine, (str, pathlib.PurePath)):
        raise ValueError('Incorrect MRI affine input')

    #----------------------------------
    # > sort out output
    if outpath is None:
        opth = fpet.parent/'VR_output'
    else:
        opth = outpath
    nimpa.create_dir(opth)

    if outfref is None:
        outfref = os.path.join(str(opth),'VRimref')

    #----------------------------------



    if fref is None:
        SZ_VX = ref_voxsize
        SZ_IM = ref_imsize
        B = np.diag(np.array([-SZ_VX, SZ_VX, SZ_VX, 1]))
        B[0, 3] = .5 * SZ_IM * SZ_VX
        B[1, 3] = (-.5 * SZ_IM + 1) * SZ_VX
        B[2, 3] = (-.5 * SZ_IM + 1) *SZ_VX
        im = np.zeros((SZ_IM,SZ_IM,SZ_IM), dtype=np.float32)
        vxstr = str(SZ_VX).replace('.','-')+'mm'
        outfref = outfref+f'_{SZ_IM}-{vxstr}.nii.gz'
        nimpa.array2nii(im, B, outfref)
        fref = outfref

    elif os.path.isfile(fref):
        log.info('using reference file: '+str(fref))
        vxstr = ''
        refd = nimpa.getnii(fref, output='all')
        SZ_VX = max(refd['voxsize'])
        SZ_IM = max(refd['dims'])
        vxstr = str(SZ_VX).replace('.','-')+'mm'

    else:
        raise ValueError('unknown reference for resampling!')


    fpetr = nimpa.resample_spm(fref, fpet, pet_affine, intrp=intrp, fimout=vt_opth/f'PET_{SZ_IM}_{vxstr}{fcomment}.nii.gz',
                                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

    fmrir = nimpa.resample_spm(fref, fmri, mri_affine, intrp=intrp, fimout=vt_opth/f'MRI_{SZ_IM}_{vxstr}{fcomment}.nii.gz',
                                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)


    return dict(fpet=fpetr, fmri=fmrir)





# =====================================================================
