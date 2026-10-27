'''
Aligning tools for PET dynamic frames
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2023"

import logging
import shutil, os
from pathlib import Path

import numpy as np
from niftypet import nimpa
import spm12

from .align import align_frames
from .proc import pvc_reblur

from scipy import ndimage

log = logging.getLogger(__name__)



def align_break_petct(niidat, cts, Cnt, qcpth=None, refpetidx=None, segment_ct=False, use_stored=False):
    '''
    Align PET images in break dynamic acquisition
    using the CT for aligning all frames within each acquisition
    followed by CT-to-CT registration and overall alignment.
    Arguments:
    ----------
    niidat:     dictionary of PET NIfTIs and DICOM info for each frame
    cts:        sorted list of CT scans for each acquisition 
    Cnt:        Constants for processing data
    refpetidx:  reference PET frame indices if intended to use PET
                frames for each acquisition instead of CTs.  CTs
                are used only for the alignment between the two
                break acquisitions. For example, when using the first
                two or last two PET frames, the call would be:
                refpetidx=[0,0] or refpetidx=[-1,-1], respectively;
                if refpetidx=None (default), then CT acquisitions are
                used as reference instead.
    segment_ct: if True, segment the CT in the aligned PET spaces for each 
                acquisition.
    '''


    # > output path from the input dictionary
    opth = niidat['outpath'].parent

    # > output path for alignment of CTs
    algnFpth = opth/(f'aligned_full')
    affsF = algnFpth/'combined_affines'
    
    nimpa.create_dir(algnFpth)
    nimpa.create_dir(affsF)

    # > output dictionary file
    fout = algnFpth/'aligned_full_output.npy'
    if use_stored and fout.is_file():
        outdct = np.load(fout, allow_pickle=True)
        outdct = outdct.item()
        return outdct
    else:
        # > output dictionary
        outdct = {}

    # > number of frames for break dynamic acquisitions A and B
    nfrmA = len(niidat['series'][0])
    nfrmB = len(niidat['series'][1])

    outdct['nfrm_t'] = (nfrmA, nfrmB)

    # > sort out the correct index for PET reference frame
    if refpetidx is not None:
        if any([refpetidx[0]<-nfrmA, refpetidx[1]<-nfrmB, refpetidx[0]>=nfrmA, refpetidx[1]>=nfrmB]):
            raise ValueError('wrong selection of PET reference frame - outside of range')

        if refpetidx[0]<0:
            refpetidx[0] += nfrmA

        if refpetidx[1]<0:
            refpetidx[1] += nfrmB


    # > DECAY CORRECTION (if requested)
    if Cnt['align']['decay_corr']:
        
        # > get the start time of each series for decay correction if requested
        ts = [sri['time'][0] for sri in niidat['descr']]
        # > index the earliest ref time
        i_tref = np.argmin(ts)
        idxs = list(range(len(ts)))
        idxs.pop(i_tref)
        i_tsrs = idxs
        # > time difference
        td = [ts[i] - ts[i_tref] for i in i_tsrs]
        if len(td) > 1:
            raise ValueError(
                'currently only one dynamic break is allowed - detected more than one')
        else:
            td = td[0]

        # > what tracer / radionuclide is used?
        istp = 'F18' * (niidat['tracer'] in Cnt['tracer']['f18']) + 'C11' * (niidat['tracer'] in Cnt['tracer']['c11'])

        # > decay constant using half-life
        lmbd = np.log(2) / nimpa.resources.riLUT[istp]['thalf']

        # > decay correction factor
        dcycrr = np.exp(lmbd * td)
    else:
        dcycrr = 1.
        
    outdct['dcycrr'] = dcycrr


    # > reference images based on CT
    fctref = [None, None]
    fpetref = [None, None]

    # > CT segmentation
    fctseg = [None, None]

    # > sum PET images
    fsum = [None, None]

    # > alignment results
    algn_frm = [None, None]

    

    if qcpth is None:
        qcpth = opth/'QC-output'
    nimpa.create_dir(qcpth)

    nacq = len(niidat['series'])
    if nacq!=2:
        raise ValueError(f'The number of acquisitions (in niidat) should be two, but given {nacq}')

    for i_acq, acq in enumerate(niidat['series']):

        # > output path for part alignment
        algnpth = opth/('aligned_acq-{}'.format)(i_acq+1)
        nimpa.create_dir(algnpth)
        
        #--------------------------------------------
        # > input PET frames
        imfrms = [acq[f]['fnii'] for f in acq]
        imtime = niidat['descr'][i_acq]['timings']
        #--------------------------------------------

        #--------------------------------------------
        # > CT reference and CoM correction for registration
        mudct = nimpa.getnii(cts[i_acq], output='all')
        mu = nimpa.ct2mu(mudct['im'])
        mu[np.isnan(mu)] = 0
        fmu = algnpth/('ct2mu_{}.nii.gz'.format(i_acq+1))
        nimpa.array2nii(mu, mudct['affine'], fmu, trnsp=mudct['transpose'], flip=mudct['flip'])

        # # > mu-map as reference in PET space
        # fres = nimpa.resample_spm(
        #     imfrms[0], fmu, np.eye(4), fimout=algnpth/(cts[i_acq].name.split('.nii')[0]+'_inPET.nii.gz'),
        #     del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
        # #~~~~~~~~~~~~~~~~~~~~
        # # > get rid of any NaNs
        # imd = nimpa.getnii(fres, output='all')
        # os.remove(fres)
        # im = imd['im']
        # im[np.isnan(im)] = 0
        # nimpa.array2nii(im, imd['affine'], fres, trnsp=imd['transpose'], flip=imd['flip'])
        # fres = Path(fres)
        # #~~~~~~~~~~~~~~~~~~~~
        # # > CT centre of mass (CoM)
        # fctcom = fres.parent/(fres.name.split('.nii')[0]+'_CoM-modified.nii')
        # dctcom = nimpa.centre_mass_corr(fres, fout=fctcom)

        # > CT centre of mass (CoM)
        fctcom = fmu.parent/(fmu.name.split('.nii')[0]+'_CoM-modified.nii')
        dctcom = nimpa.centre_mass_corr(fmu, fout=fctcom)
        log.info(f'Modified CoM (CT): {fctcom}')

        # > CT reference image for CT registration and optionally PET
        fctref[i_acq] = fctcom

        #~~~~~~~~~~~~~~~~~~~~~~~~~~
        if segment_ct:
            # > CT-based segmentations
            ctseg_out = opth/'CT_seg'
            nimpa.create_dir(ctseg_out)

            # > get rid of head holder
            frgrnd = mudct['im']>-500
            frgrnd = ndimage.binary_closing(frgrnd, iterations=2)
            frgrnd = ndimage.binary_fill_holes(frgrnd)
            lbl, n = ndimage.label(frgrnd)
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            hd_msk = (lbl == sizes.argmax())

            # > soft tissue and skull
            sft = hd_msk * (mudct['im']>-650) * (mudct['im']<300)
            skl = hd_msk * (mudct['im']>300)

            ct_seg = 1.0*sft + 2.0*skl

            fct_seg = ctseg_out/'temp_ct_seg.nii.gz'
            nimpa.array2nii(ct_seg, mudct['affine'], ctseg_out/'temp_ct_seg.nii.gz', trnsp=mudct['transpose'], flip=mudct['flip'])

            # fseg_pet = nimpa.resample_spm(
            #     imfrms[0], fct_seg, np.eye(4), fimout=ctseg_out/('CT_seg_part-{}_inPET.nii.gz'.format(i_acq+1)),
            #     del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

            fsegcom = ctseg_out/('CT_seg_part-{}_CoM.nii.gz'.format(['A','B'][i_acq]))
            nimpa.centre_mass_corr(fct_seg, fout=fsegcom, com=dctcom['com_abs'])

            fctseg[i_acq] = fsegcom

            #matshow(mudct['im'][76,...], cmap='bone')
            #matshow(ct_seg[76,...], cmap='gray')
            #nimpa.imscroll([ct_seg, mudct['im']], cmap='gray')
            #matshow(sft[76,...], cmap='gray')
            #matshow(skl[76,...], cmap='gray')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~
        #--------------------------------------------

        #--------------------------------------------
        # > pick the reference for the two break PET dynamic acquisitions
        if refpetidx is None:
            fpetref[i_acq] = fctcom
        else:
            if refpetidx[i_acq] in range(len(niidat['series'][i_acq])):
                #> get the name of the selected reference frame
                fnm = niidat['descr'][i_acq]['frms'][refpetidx[i_acq]]
                fpref = niidat['series'][i_acq][fnm]['fnii']
                fpetcom = fpref.parent/(fpref.name.split('.nii')[0]+'_CoM-modified.nii')
                nimpa.centre_mass_corr(fpref, fout=fpetcom)
                log.info(f'Modified CoM (PET): {fpetcom}')
                fpetref[i_acq] = fpetcom
            else:
                raise ValueError('incorrect frame index for PET reference frame!')

        #--------------------------------------------

        #--------------------------------------------
        # > ALIGNMENT TO REFERENCE
        algn_frm[i_acq] = align_frames(imfrms, imtime, fpetref[i_acq], Cnt,
            reg_tool='spm', spm_com_corr=True, outpath=algnpth)
        #--------------------------------------------
        
        #--------------------------------------------
        # > PET sum of aligned frames
        refdct = nimpa.getnii(fpetref[i_acq], output='all')
        fsum[i_acq] = qcpth/('PET_aligned_avg_part{}.nii.gz'.format('A'*(i_acq==0)+'B'*(i_acq==1)))
        nimpa.array2nii(
            np.sum(algn_frm[i_acq]['im4d'],axis=0)/algn_frm[i_acq]['im4d'].shape[0],
            refdct['affine'],
            fsum[i_acq],
            trnsp=refdct['transpose'], flip=refdct['flip'])
        #--------------------------------------------

    #--------------------------------------------
    # > co-register CTs with the second (as in static imaging) acting as the reference
    ct_reg = nimpa.coreg_spm(fctref[1], fctref[0], outpath=algnFpth,  costfun='nmi', fwhm_ref=2., fwhm_flo=2.)

    frct = nimpa.resample_spm(fctref[1], fctref[0], ct_reg['affine'],
        fimout=algnFpth/(fctref[0].name.split('.nii')[0]+'_registered-to-CT-B.nii.gz'),
        del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
    #--------------------------------------------

    #--------------------------------------------
    # > CT registration to spaces A and B ()
    petct_Areg = None
    petct_Breg = None
    # > When PET frames are used as reference for alignment
    if refpetidx is not None:
        # > register CT to average PET frames
        petct_Areg = nimpa.coreg_spm(
            fctref[0],
            fsum[0],
            outpath=algnFpth,
            fcomment='_petct_partA',
            costfun='nmi',
            fwhm_ref=2., fwhm_flo=2.)

        petct_Breg = nimpa.coreg_spm(
            fsum[1],
            fctref[1],
            outpath=algnFpth,
            fcomment='_petct_partB',
            costfun='nmi',
            fwhm_ref=2., fwhm_flo=2.)

        faligned = []
        faffines = []

        for fi, f in enumerate(algn_frm[0]['fnii_com']):
            print(f)

            # > get the affine from the first registration to the appropriate reference PET frame
            affF = np.loadtxt(algn_frm[0]['affines'][fi])

            # > get it to CT space in acquisition part A
            affF = np.matmul(affF, petct_Areg['affine'])

            # > combine the affine with the CT-to-CT registration affine
            affF = np.matmul(affF, ct_reg['affine'])

            # > and now add the CT-to-PET part B
            affF = np.matmul(affF, petct_Breg['affine'])

            faff = affsF/(Path(f).name.split('.nii')[0]+'_affine.txt')
            np.savetxt(faff, affF)
            faffines.append(faff)

            # > resample the frame to the overall alignment
            frpet = nimpa.resample_spm(fpetref[1], f, affF,
                fimout=algnFpth/(f.name.split('.nii')[0]+'_full_reg.nii.gz'),
                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

            faligned.append(Path(frpet))

        # > the second part of the acquisition (is already in the CT space of interest)
        keys2 = list(niidat['series'][1].keys())
        imdct = nimpa.getnii(algn_frm[1]['faligned'][0], output='all')
        imur = np.zeros( imdct['shape'], dtype=np.float32)

        for fi, f in enumerate(algn_frm[1]['faligned']):
            print(f)
            
            # > correction for decay
            if Cnt['align']['decay_corr']:
                fdcrr = algnFpth/(Path(f).name.split('.nii')[0]+'_dcrr.nii.gz')
                imd = nimpa.getnii(f, output='all')
                nimpa.array2nii(
                    imd['im']*dcycrr,
                    imd['affine'],
                    fdcrr,
                    trnsp=imd['transpose'],
                    flip=imd['flip'])
                faligned.append(fdcrr)
            else:
                fcopy = algnFpth/Path(f).name
                shutil.copyfile(f, fcopy)
                faligned.append(fcopy)

            faff = affsF/(Path(f).name.split('.nii')[0]+'_affine.txt')
            shutil.copyfile(algn_frm[1]['affines'][fi], faff)
            faffines.append(faff)

            # > create a single uptake ratio (UR) image
            imur += nimpa.getnii(f)

        fur = algnFpth/'UR-PET-aligned_partB.nii.gz'
        nimpa.array2nii(imur, imdct['affine'],fur, trnsp=imdct['transpose'], flip=imdct['flip'])

    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # > When CT is used for reference
    else:
        # > the first part of the acquisition
        # > the index of acquisition to be aligned (first)
        i_acq = 0
        faligned = []
        faffines = []

        for fi, f in enumerate(algn_frm[i_acq]['fnii_com']):
            print(f)

            # > get the affine from the first registration to the appropriate CT
            aff0 = np.loadtxt(algn_frm[i_acq]['affines'][fi])

            # > combine the affine with the CT-to-CT registration affine
            #affF = np.matmul(ct_reg['affine'], aff0)
            affF = np.matmul(aff0, ct_reg['affine'])

            faff = affsF/(Path(f).name.split('.nii')[0]+'_affine.txt')
            np.savetxt(faff, affF)
            faffines.append(faff)

            # > resample the frame to the overall alignment
            frpet = nimpa.resample_spm(fctref[1], f, affF,
                fimout=algnFpth/(f.name.split('.nii')[0]+'_full_reg.nii.gz'),
                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

            faligned.append(Path(frpet))

        # > the second part of the acquisition (is already in the CT space of interest)
        i_acq = 1
        keys2 = list(niidat['series'][i_acq].keys())
        imdct = nimpa.getnii(algn_frm[i_acq]['faligned'][0], output='all')
        imur = np.zeros( imdct['shape'], dtype=np.float32)

        for fi, f in enumerate(algn_frm[i_acq]['faligned']):
            print(f)

            # > correction for decay
            if Cnt['align']['decay_corr']:
                fdcrr = algnFpth/(Path(f).name.split('.nii')[0]+'_dcrr.nii.gz')
                imd = nimpa.getnii(f, output='all')
                nimpa.array2nii(
                    imd['im']*dcycrr,
                    imd['affine'],
                    fdcrr,
                    trnsp=imd['transpose'],
                    flip=imd['flip'])
                faligned.append(fdcrr)
            else:
                fcopy = algnFpth/Path(f).name
                shutil.copyfile(f, fcopy)
                faligned.append(fcopy)


            faff = affsF/(Path(f).name.split('.nii')[0]+'_affine.txt')
            shutil.copyfile(algn_frm[i_acq]['affines'][fi], faff)
            faffines.append(faff)

            # > create a single uptake ratio (UR) image
            imur += nimpa.getnii(f)

        fur = opth/'UR-PET-aligned.nii.gz'
        nimpa.array2nii(imur, imdct['affine'],fur, trnsp=imdct['transpose'], flip=imdct['flip'])
    #--------------------------------------------


    #--------------------------------------------
    imdct = nimpa.getnii(faligned[0], output='all')

    im4d = np.zeros( (len(faligned),) + imdct['shape'], dtype=np.float32)
    im4A = np.zeros( (nfrmA,) + imdct['shape'], dtype=np.float32)
    im4B = np.zeros( (nfrmB,) + imdct['shape'], dtype=np.float32)
    imsum = np.zeros(imdct['shape'], dtype=np.float32)

    # > combine all images into one 4D PET NIfTI and two 4D dynamic break acquisitions
    for fi, f in enumerate(faligned):
        print('> saving frames to 4D NIfTI images...')
        print(f)
        im4d[fi,...] = nimpa.getnii(f)
        if fi<nfrmA:
            im4A[fi,...] = nimpa.getnii(f)
        else:
            im4B[fi-nfrmA,...] = nimpa.getnii(f)
        imsum += nimpa.getnii(f)

    f4d = algnFpth/'PET-4D-aligned.nii.gz'
    f4A = algnFpth/'PET-4D-aligned_break-A.nii.gz'
    f4B = algnFpth/'PET-4D-aligned_break-B.nii.gz'
    fsumF = opth/'PET-summed-aligned.nii.gz'
    nimpa.array2nii(im4d, imdct['affine'], f4d, trnsp=imdct['transpose'], flip=imdct['flip'])
    nimpa.array2nii(im4A, imdct['affine'], f4A, trnsp=imdct['transpose'], flip=imdct['flip'])
    nimpa.array2nii(im4B, imdct['affine'], f4B, trnsp=imdct['transpose'], flip=imdct['flip'])
    nimpa.array2nii(imsum, imdct['affine'],fsumF, trnsp=imdct['transpose'], flip=imdct['flip'])
    #--------------------------------------------


    outdct['align_acq'] = algn_frm
    outdct['ctref'] = fctref
    outdct['petref'] = fpetref
    outdct['fsumAB'] = fsum
    outdct['ct_reg'] = ct_reg
    outdct['petct_aff_A'] = petct_Areg
    outdct['petct_aff_B'] = petct_Breg
    outdct['fct_reg'] = frct
    outdct['fur'] = fur
    outdct['fsum'] = fsumF
    outdct['faligned'] = faligned
    outdct['fpet4d'] = f4d
    outdct['fpet4A'] = f4A
    outdct['fpet4B'] = f4B
    outdct['faffines'] = faffines
    outdct['fnii_com'] = algn_frm[0]['fnii_com'] + algn_frm[1]['fnii_com']
    #--------------------------------------
    # > for CT segmentation
    if segment_ct:        
        M_A = np.loadtxt(petct_Areg['faff'])
        M_Ainv = np.linalg.inv(M_A)
        M_B = np.loadtxt(petct_Breg['faff'])

        fpetA_avg = fsum[0]
        fpetB_avg = fsum[1]

        fsegA = ctseg_out/(fctseg[0].name.split('.nii')[0]+'__aligned.nii.gz')
        fsegB = ctseg_out/(fctseg[1].name.split('.nii')[0]+'__aligned.nii.gz')

        fsegA_ = nimpa.resample_spm(fpetA_avg, fctseg[0], M_Ainv, intrp=0, fimout=fsegA,
                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
        fsegB_ = nimpa.resample_spm(fpetB_avg, fctseg[1], M_B, intrp=0, fimout=fsegB,
                del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

        outdct['ct_seg'] = [fsegA, fsegB] #fctseg
        # > CT segmentations in the CT spaces (for bone correction in native PET spaces)
        outdct['ct_seg_ct'] = fctseg
    #--------------------------------------
        

    if use_stored:
        np.save(fout, outdct)

    return outdct




#=====================================================================================

def bone_corr(fpet, fseg, fwhm=(6.4, 4.5, 4.5), itr=5, dev_id=0, outpath=None, outpath_bc=None):
    '''
    Correct PET for bone (skull) uptake using the CT segmentation:
    (1) iterative Yang PVC with CT regions (PVC_CT), (2) removal of
    the bone signal and (3) re-blurring across but not within regions
    (PVC_inv).
    Arguments:
    ----------
    fpet:       NIfTI file of the PET image to be corrected
    fseg:       NIfTI file of the CT segmentation (0 = air, 1 = soft
                tissue, 2 = bone) in the same space as `fpet`
    fwhm:       FWHM [mm] of the scanner PSF in (z, y, x)
    itr:        number of iterations of the iterative Yang PVC
    dev_id:     GPU device ID (`False` forces CPU)
    outpath:    output folder for the PVC image (and the bone-corrected
                image if `outpath_bc` is not given); by default
                `bone_corr` subfolder next to `fpet`
    outpath_bc: output folder for the bone-corrected image; by default
                the same as `outpath`
    '''

    # > segmentation image based on the CT: 0 = air, 1 = soft tissue, 2 = bone
    dsegct = nimpa.getnii(fseg, output='all')
    segct = dsegct['im']

    # > NaN / inf -> air (0)
    bad = ~np.isfinite(segct)
    if bad.any():
        log.warning("%d NaN/inf voxels in CT segmentation: set to 0 (air)", bad.sum())
        segct[bad] = 0

    # > non-integer labels -> rounded to nearest integer
    nonint = segct != np.round(segct)
    if nonint.any():
        log.warning("%d non-integer labels in CT segmentation: rounded "
                    "(better to resample with nearest neighbour, intrp=0)", nonint.sum())
        segct = np.round(segct)

    # > labels outside 0..2 -> clipped to the nearest valid label
    labels = np.unique(segct)
    if not set(labels) <= {0, 1, 2}:
        log.warning("unexpected labels in CT segmentation %s: clipped to 0..2", labels)
        segct = np.clip(segct, 0, 2)

    segct = segct.astype(np.int32)

    #-----------------------------------
    # > PET image
    dpet = nimpa.getnii(fpet, output='all')
    pet = np.nan_to_num(dpet['im'])

    # > PET and segmentation must be in the same space
    if pet.shape != segct.shape:
        raise IndexError(f'PET {pet.shape} and CT segmentation {segct.shape} do not match')
    if not np.allclose(dpet['affine'], dsegct['affine'], atol=1e-3):
        raise ValueError('PET and CT segmentation are not in the same space (different affines)')

    # > scanner PSF (z, y, x)
    psf = nimpa.psf_gaussian(vx_size=dpet['voxsize'], fwhm=fwhm)

    # > 1. PVC_CT (iterative Yang, CT regions)
    petpvc, m_a = nimpa.iyang(pet, psf, segct, {'DEVID': dev_id}, itr=itr)

    # > 2. bone subtraction: f_s = (1 - I_b) * f_pvc
    bone = segct == 2
    petsub = petpvc * (~bone)

    # > 3. PVC_inv: re-blur across but not within regions
    petbc = pvc_reblur(petsub, segct, psf, dev_id=dev_id)
    #-----------------------------------

    # > save the intermediate images in the PET geometry
    if outpath is None:
        outpath = Path(fpet).parent/'bone_corr'
    opth = Path(outpath)
    nimpa.create_dir(opth)
    opth_bc = opth if outpath_bc is None else Path(outpath_bc)
    nimpa.create_dir(opth_bc)

    fnm = Path(fpet).name.split('.nii')[0]

    # > PVC_CT image (before bone removal)
    fpvc = opth/(fnm+'_pvc-ct.nii.gz')
    nimpa.array2nii(petpvc, dpet['affine'], fpvc, storage_as=dpet)
    log.info(f'PVC (CT regions) PET saved to: {fpvc}')

    # > bone-corrected image
    fbc = opth_bc/(fnm+'_bone-corrected.nii.gz')
    nimpa.array2nii(petbc, dpet['affine'], fbc, storage_as=dpet)
    log.info(f'bone-corrected PET saved to: {fbc}')

    return {
        'fpetbc': fbc, 'petbc': petbc, 'fpetpvc': fpvc, 'petpvc': petpvc, 'm_a': m_a,
        'psf': psf}




def bone_corr_dyn(aligned_ct, fwhm=(6.4, 4.5, 4.5), itr=5, dev_id=0, outpath=None,
                  use_stored=False, del_intermediate=False):
    '''
    Bone correction of dynamic break PET frames, each in its native
    PET space, followed by the same alignment as in `align_break_petct`,
    so that the output is in the space of the fully aligned PET.
    For each frame the CT segmentation (in its CT space) is resampled
    to the native frame, the frame is bone corrected (`bone_corr`) and
    then transformed as the uncorrected frame:
      A: the combined affine (frame -> PET-A ref -> CT A -> CT B -> PET-B ref),
      B: the frame affine (frame -> PET-B ref) and decay correction.
    Arguments:
    ----------
    aligned_ct: output dictionary of `align_break_petct` run with
                `segment_ct=True` and PET reference frames (`refpetidx`)
    fwhm:       FWHM [mm] of the scanner PSF in (z, y, x)
    itr:        number of iterations of the iterative Yang PVC
    dev_id:     GPU device ID (`False` forces CPU)
    outpath:    output folder; by default the folder of the summed
                aligned PET
    use_stored: if True, use the stored output dictionary if present
    del_intermediate: if True, delete the intermediate folder with the
                native CT segmentations and the native PVC frames,
                after all outputs have been saved; the native bone-
                corrected frames (`bcorrected`) are kept
    '''

    if 'ct_seg' not in aligned_ct:
        raise ValueError('no CT segmentation: run `align_break_petct` with segment_ct=True')
    if aligned_ct['petct_aff_A'] is None or aligned_ct['petct_aff_B'] is None:
        raise NotImplementedError('bone correction is only available for alignment with'
                                  ' PET reference frames (`refpetidx`)')

    # > output paths
    if outpath is None:
        outpath = Path(aligned_ct['fsum']).parent
    opth = Path(outpath)/'bone_corr'
    # > intermediate: native CT segmentations and native PVC frames
    intpth = opth/'intermediate'
    segpth = intpth/'CT_seg_native'
    pvcpth = intpth/'PVC_native'
    # > native bone-corrected frames (input for further processing, e.g., MNI)
    bcpth = opth/'bcorrected'
    algnpth = opth/'aligned'
    for p in (opth, intpth, segpth, pvcpth, bcpth, algnpth):
        nimpa.create_dir(p)

    # > output dictionary file
    fout = opth/'bone_corr_dyn_output.npy'
    if use_stored and fout.is_file():
        return np.load(fout, allow_pickle=True).item()

    # > CT segmentations in the CT spaces of acquisitions A and B
    if 'ct_seg_ct' in aligned_ct:
        fctseg = aligned_ct['ct_seg_ct']
    else:
        fctseg = [Path(f).parent/(Path(f).name.split('__aligned')[0]+'.nii.gz')
                  for f in aligned_ct['ct_seg']]

    # > PET-A -> CT-A and CT-B -> PET-B affines
    M_A = aligned_ct['petct_aff_A']['affine']
    M_B = aligned_ct['petct_aff_B']['affine']

    # > reference of the full alignment (PET-B reference frame)
    fref = aligned_ct['petref'][1]
    dcycrr = aligned_ct['dcycrr']
    nfrmA = aligned_ct['nfrm_t'][0]

    # > PVC (native), bone-corrected (native) and fully aligned frames
    fpvc_nat = []
    fbc_nat = []
    fbc_algn = []
    # > native segmentations, one for each frame
    fseg_nat = []

    for i_acq in range(2):
        algn = aligned_ct['align_acq'][i_acq]
        if algn['fnii_com'] is None:
            raise ValueError('frames have to be aligned with CoM correction (spm_com_corr=True)')

        # > native segmentations per affine (frames in a mashed group share it)
        segs = {}

        for fi, f in enumerate(algn['fnii_com']):
            log.info(f'bone correction of frame: {f}')

            # > frame -> PET reference affine
            S = np.loadtxt(algn['affines'][fi])

            #-----------------------------------
            # > CT segmentation in the native frame space
            if algn['affines'][fi] not in segs:
                if i_acq == 0:
                    # > frame -> PET-A ref -> CT A
                    M = np.linalg.inv(np.matmul(S, M_A))
                else:
                    # > frame -> PET-B ref <- CT B
                    M = np.matmul(M_B, np.linalg.inv(S))

                fseg = segpth/(fctseg[i_acq].name.split('.nii')[0]+'__'+Path(f).name.split('.nii')[0]+'.nii.gz')
                nimpa.resample_spm(f, fctseg[i_acq], M, intrp=0, fimout=fseg,
                    del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
                segs[algn['affines'][fi]] = fseg

            fseg_nat.append(segs[algn['affines'][fi]])
            #-----------------------------------

            # > bone correction in the native frame space
            bc = bone_corr(f, fseg_nat[-1], fwhm=fwhm, itr=itr, dev_id=dev_id,
                           outpath=pvcpth, outpath_bc=bcpth)
            fpvc_nat.append(bc['fpetpvc'])
            fbc_nat.append(bc['fpetbc'])

            #-----------------------------------
            # > the same transformation as for the uncorrected frames
            fbc = bc['fpetbc']
            fnm = fbc.name.split('.nii')[0]
            if i_acq == 0:
                frs = nimpa.resample_spm(fref, fbc, np.loadtxt(aligned_ct['faffines'][fi]),
                    fimout=algnpth/(fnm+'_full_reg.nii.gz'),
                    del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
                fbc_algn.append(Path(frs))
            else:
                if algn['modified'][fi]:
                    frs = nimpa.resample_spm(fref, fbc, S, intrp=1.,
                        fimout=algnpth/(fnm+'_reg.nii.gz'),
                        del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
                else:
                    # > not moved in the alignment (below the motion threshold)
                    frs = fbc

                # > correction for decay (dcycrr = 1 if not requested)
                fdcrr = algnpth/(fnm+'_dcrr.nii.gz')
                imd = nimpa.getnii(frs, output='all')
                nimpa.array2nii(imd['im']*dcycrr, imd['affine'], fdcrr, storage_as=imd)
                fbc_algn.append(fdcrr)
            #-----------------------------------

    #--------------------------------------------
    # > combine all frames into one 4D PET NIfTI and two 4D dynamic break acquisitions
    imdct = nimpa.getnii(fbc_algn[0], output='all')
    im4d = np.zeros((len(fbc_algn),) + imdct['shape'], dtype=np.float32)
    for fi, f in enumerate(fbc_algn):
        im4d[fi, ...] = nimpa.getnii(f)
    imsum = np.sum(im4d, axis=0)

    f4d = opth/'PET-4D-aligned_bone-corrected.nii.gz'
    f4A = opth/'PET-4D-aligned_break-A_bone-corrected.nii.gz'
    f4B = opth/'PET-4D-aligned_break-B_bone-corrected.nii.gz'
    fsum = opth/'PET-summed-aligned_bone-corrected.nii.gz'
    nimpa.array2nii(im4d, imdct['affine'], f4d, storage_as=imdct)
    nimpa.array2nii(im4d[:nfrmA], imdct['affine'], f4A, storage_as=imdct)
    nimpa.array2nii(im4d[nfrmA:], imdct['affine'], f4B, storage_as=imdct)
    nimpa.array2nii(imsum, imdct['affine'], fsum, storage_as=imdct)
    #--------------------------------------------

    outdct = {
        'fbc_native': fbc_nat, 'faligned': fbc_algn, 'fpet4d': f4d, 'fpet4A': f4A,
        'fpet4B': f4B, 'fsum': fsum}

    # > intermediate files (deleted only after all outputs are saved)
    if del_intermediate:
        shutil.rmtree(intpth)
        log.info(f'deleted intermediate files: {intpth}')
    else:
        outdct['fseg_native'] = fseg_nat
        outdct['fpvc_native'] = fpvc_nat

    if use_stored:
        np.save(fout, outdct)

    return outdct
