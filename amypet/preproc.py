'''
Preprocessing tools for AmyPET core processes
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022"

import logging as log
import os
import shutil
from datetime import datetime
from pathlib import Path
from subprocess import run
import copy

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
def explore_indicom(input_fldr, tracer=None, suvr_win_def=None, outpath=None, margin=0.1):
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


    # =================== SUVr =======================
    def find_suvr(t_frms_, msrs_class_, t_starts_, t_stops_):
        ''' find SUVr frames among the static/dynamic frames
        '''

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

        if t_frms_[0][0] < suvr_win[0] + mrgn_suvr_start and \
            t_frms_[0][0] > suvr_win[0] - mrgn_suvr_start and \
            acq_dur > suvr_twindow[tracer][2] - mrgn_suvr_dur:

            t0_suvr = min(t_starts_, key=lambda x: abs(x - suvr_win[0]))
            t1_suvr = min(t_stops_, key=lambda x: abs(x - suvr_win[1]))

            frm_0 = t_starts_.index(t0_suvr)
            frm_1 = t_stops_.index(t1_suvr)

            # > SUVr frame range
            suvr_frm_range = range(frm_0, frm_1 + 1)

            # > indicate which frames were selected for SUVr relative to all static/dynamic frames
            frms_sel = [i in suvr_frm_range for i, s in enumerate(srs_t)]

            msrs_class_[-1]['acq'].append('suvr')
            msrs_class_[-1]['suvr'].update({
                'time': (t0_suvr, t1_suvr),
                'timings': [t_frms_[f] for f in range(frm_0, frm_1 + 1)],
                'idxs': (frm_0, frm_1),
                'frms': [s for i, s in enumerate(srs_t) if i in suvr_frm_range],
                'frms_sel': frms_sel})
        else:
            log.warning('The acquisition does not cover the requested time frame!')
    # ===================================================


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
            msrs_class.append({
                    'acq': [acq_type],
                    'time': (t_starts[0], t_stops[-1]),
                    'timings': t_frms,
                    'idxs': (0, len(t_frms) - 1),
                    'frms': [s for i, s in enumerate(srs_t)],
                    'suvr':{}})

            find_suvr(t_frms, msrs_class, t_starts, t_stops)
        
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
            t1_dyn = min(t_stops, key=lambda x: (x - fulldyn_time))

            frm_0 = t_starts.index(t0_dyn)
            frm_1 = t_stops.index(t1_dyn)

            msrs_class.append({
                'acq': [acq_type],
                'time': (t0_dyn, t1_dyn),
                'timings': t_frms,
                'idxs': (frm_0, frm_1),
                'frms': [s for i, s in enumerate(srs_t) if i in range(frm_0, frm_1 + 1)],
                'suvr':{}})

            find_suvr(t_frms, msrs_class, t_starts, t_stops)


    return {'series': msrs_t, 'descr': msrs_class, 'outpath': amyout, 'tracer': tracer}



# =====================================================================
def convert2nii(indct, outpath=None):
    '''convert the individual DICOM series to NIfTI
       and output a new updated NIfTI dictionary
    '''


    if 'series' not in indct:
        raise ValueError('The key <series> is not present in the input dictionary!')
   
    # > number of studies in  the folder
    Sn = len(indct['series'])

    # > sort out the output path
    if outpath is not None:
        niidir = Path(outpath)
    else:
        outpath = Path(indct['outpath']).parent
        niidir = outpath/'NIfTIs'
    nimpa.create_dir(niidir)


    # > remove any files from previous runs
    files = niidir.glob('*')
    for f in files:
        if f.is_file():
            os.remove(f)
        else:
            shutil.rmtree(f)
    

    niidat = copy.deepcopy(indct)
    niidat['outpath'] = niidir

    for sti in range(Sn):
        for k in indct['series'][sti]:
            run([dcm2niix.bin, '-i', 'y', '-v', 'n', '-o', niidir, 'f', '%f_%s',
                 indct['series'][sti][k]['files'][0].parent])

            # > get the converted NIfTI file
            fnii = list(niidir.glob(str(indct['series'][sti][k]['tacq']) + '*.nii*'))
            if len(fnii) != 1:
                raise ValueError('Unexpected number of converted NIfTI files')
            else:
                niidat['series'][sti][k].pop('files', None)
                niidat['series'][sti][k]['fnii'] = fnii[0]

    return niidat
# =====================================================================





# ========================================================================================
def id_acq(dctdat, acq_type='suvr', output_series_id=False):
    '''
    Identify acquisition type such as SUVr or coffee break data 
    in the dictionary of classified DICOM/NIfTI series.
    Arguments:
    - dctdat:       the dictionary with all DICOM/NIfTI series 
                    (e.g., as the output of explore_input).
    - acq_type:     can be 'suvr' or 'break'/'breakdyn' or 'dyn'/'fulldyn'.
    '''

    if acq_type in ['break', 'breakdyn']:
        acq = 'breakdyn'
    elif acq_type in ['dyn', 'fulldyn']:
        acq = 'fulldyn'
    elif acq_type in ['suvr']:
        acq = 'suvr'
    else:
        raise ValueError('unrecognised acquisition type!')

    # > find the SUVr-compatible acquisition and its index
    acq_find = [(i,a) for i,a in enumerate(dctdat['descr']) if acq in a['acq']]

    if len(acq_find)>1:
        raise ValueError('too many SUVr/static DICOM series detected: only one is accepted')
    elif len(acq_find)==0:
        log.info('no fully dynamic data found.')
        return None
    else:
        acq_find = acq_find[0]

        # > time-sorted data for SUVr
        tdata = dctdat['series'][acq_find[0]]

        # > data description with classification
        tdata['descr'] = acq_find[1]

    if output_series_id:
        return (acq_find[0], tdata)
    else:
        return tdata
# ========================================================================================





# ========================================================================================
def rem_artefacts(niidat, artefact='endfov', frames=None, zmrg=10):
    ''' Remove artefacts from NIfTI images.
        Arguments:
        - niidat:       dictionary of all series NIfTI data
        - artefact:     what kind of artefact (default is the end 
                        of FOV)
        - frames:       the idx of frames which should be corrected
        - zmrg:         the axial (z) voxel margin where considering
                        correction of the ends of FOV 
    '''

    # > CORRECT FOR FOV-END ARTEFACTS
    if artefact=='endfov':
    
        # > extract the early time frames data
        bdyn = id_acq(niidat, acq_type='break', output_series_id=True)
        fdyn = id_acq(niidat, acq_type='dyn', output_series_id=True)
        if not bdyn and not fdyn:
            log.info('no early dynamic data detected')
            return niidat
        else:
            if bdyn:
                si = bdyn[0]
                dyn_tdat = bdyn[1]
            else:
                si = fdyn[0]
                dyn_tdat = fdyn[1]

        
        for i,k in enumerate(dyn_tdat['descr']['frms']):
            if i in frames:
                imdct = nimpa.getnii(dyn_tdat[k]['fnii'], output='all')
                im = imdct['im']
                zprf = np.sum(im, axis=(1,2))
                ztop = np.max(zprf[:zmrg])
                zbtm = np.max(zprf[-zmrg:])
                zmid = np.max(zprf[zmrg:-zmrg])
                #print(k, ztop, zbtm, zmid, ztop>zmid, zbtm>zmid)

                # > remove the artefacts from top and bottom
                if zbtm>zmid or ztop>zmid:
                    if zbtm>zmid:
                        zidx = len(zprf)-zmrg + np.where(zprf[-zmrg:]>zmid)[0]
                        im[zidx,...] = 0
                    if ztop>zmid:
                        zidx = np.where(zprf[:zmrg]>zmid)[0]
                        im[zidx,...] = 0
                
                    # > save to a corrected image
                    fnew = dyn_tdat[k]['fnii'].parent/(dyn_tdat[k]['fnii'].name.split('.nii')[0]+'_corrected.nii')
                    nimpa.array2nii(
                        im,
                        imdct['affine'],
                        fnew,
                        descrip='AmyPET: corrected for FOV-end artefacts',
                        trnsp=(imdct['transpose'].index(0), imdct['transpose'].index(1),
                               imdct['transpose'].index(2)),
                        flip=imdct['flip'])

                    niidat['series'][si][k]['fnii'] = fnew

        return niidat
    
    else:
        raise ValueError('This artefact correction is not available at this stage.')
# ========================================================================================





# =====================================================================
def native_proc(
        cl_dct,
        atlas='aal',
        res='1',
        outpath=None,
        refvoi_idx=None,
        refvoi_name=None,
        resample_mr=False):
    '''
    Preprocess SPM GM segmentation (from CL output) and AAL atlas
    to native PET space which is trimmed and upscaled to MR resolution.

    cl_dct:     CL process output dictionary as using the normalisation
                and segmentation from SPM as used in CL.
    refvoi_idx: indices of the reference region specific for the chosen
                atlas.  The reference VOI will then be separately
                provided in the output.
    refvoi_name: name of the reference region/VOI
    resample_mr: if True resamples T1w MR image to the trimmed PET.

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
    import matlab as ml
    bbox = spm12.get_bbox(petdct)

    # > get the inverse affine transform to PET native space
    M = np.linalg.inv(cl_dct['reg2']['affine'])
    Mm = ml.double(M.tolist())

    # > copy the inverse definitions to be modified with affine to native PET space
    fmod = shutil.copyfile(cl_dct['norm']['invdef'],
                           cl_dct['norm']['invdef'].split('.')[0] + '_2nat.nii')
    eng = spm12.ensure_spm()
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

    # > resample also T1w MR image to the trimmed PET
    if resample_mr:
        ft1pet = spm12.resample_spm(trmout['ftrm'], cl_dct['reg1']['freg'], M, intrp=4.0, outpath=natout,
                    pickname='flo', fcomment='_T1w_in_PET', del_ref_uncmpr=True,
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
        intrp=1.0,
        activity=None,
        weight=None,
        ref_voxsize=1.0,
        ref_imsize=256,
        fref=None,
        outfref=None,
        outpath=None,
        fcomment=''):
    '''
    Generate PET and the accompanying MRI images for amyloid visual
    reads aligned (rigidly) to the MNI space.

    Arguments:
    - fpet:     the PET file name
    - fmri:     the MRI (T1w) file name
    - pet_affine:   PET affine given as a file path or a Numpy array 
    - mri_affine:   MRI affine given as a file path or a Numpy array 
    - intrp:    interpolation level used in resampling 
                (0:NN, 1: trilinear, etc.) 
    - activity: activity in [Bq] of the injected dose, corrected to the 
                scan start time (the time to which reconstruction 
                corrects for decay)
    - weight:   the weight in [kg] of the participant scanned
    - ref_voxsize:  the reference voxel size isotropically (default 1.0 mm)
    - ref_imsize:   the reference image size isotropically (default (256))
    - fref:     the reference image file path instead of the two above
    - outpath:  the output path
    - fcomment: the output file name suffix/comment 
    - outfref:  if reference given using `ref_voxsize` and `ref_imsize`
                instead of reference file path `ferf`, the reference
                image will be save to this path.
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
        outfref = opth

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
        outfref = outfref/f'VRimref_{SZ_IM}-{vxstr}.nii.gz'
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


    fpetr = nimpa.resample_spm(fref, fpet, pet_affine, intrp=intrp, fimout=opth/f'PET_{SZ_IM}_{vxstr}{fcomment}.nii.gz',
                                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

    fmrir = nimpa.resample_spm(fref, fmri, mri_affine, intrp=intrp, fimout=opth/f'MRI_{SZ_IM}_{vxstr}{fcomment}.nii.gz',
                                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

    out = dict(fpet=Path(fpetr), fmri=Path(fmrir), fref=fref)

    if activity is not None and weight is not None:
        # > correct the weight to grams
        weight *= 1e3
        petrd = nimpa.getnii(fpetr, output='all')
        suvim = petrd['im']/(activity/weight)

        fout = opth/f'PET-SUV_{SZ_IM}_{vxstr}{fcomment}.nii.gz'
        nimpa.array2nii(
            suvim,
            petrd['affine'],
            fout,
            trnsp = (petrd['transpose'].index(0), petrd['transpose'].index(1), petrd['transpose'].index(2)),
            flip = petrd['flip'])

        out['fsuv'] = fout

    return out





# =====================================================================
