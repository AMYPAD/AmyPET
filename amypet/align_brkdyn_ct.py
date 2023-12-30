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

log = logging.getLogger(__name__)



def align_break_petct(niidat, cts, Cnt, qcpth=None):
    '''
    Align PET images in break dynamic acquisition
    using the CT for aligning all frames within each acquisition
    followed by CT-to-CT registration and overall alignment.
    Arguemnts:
    ----------
    niidat:     dictionary of PET NIfTIs and DICOM info for each frame
    cts:        sorted list of CT scans for each acquisition 
    Cnt:        Constants for processing data
    '''

    # > reference images based on CT
    fref = [None, None]

    #fmus = [None, None]

    # > sum PET images
    fsum = [None, None]

    # > alignment results
    algn_frm = [None, None]

    opth = niidat['outpath'].parent

    # > output path for alignment of CTs
    algnFpth = opth/(f'aligned_full')
    affsF = algnFpth/'combined_affines'
    
    nimpa.create_dir(algnFpth)
    nimpa.create_dir(affsF)

    if qcpth is None:
        qcpth = opth/'QC-output'
    nimpa.create_dir(qcpth)

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
        # > CT-based PET reference and CoM correction
        mudct = nimpa.getnii(cts[i_acq], output='all')
        mu = nimpa.ct2mu(mudct['im'])
        fmu = algnpth/('ct2mu_{}.nii.gz'.format(i_acq+1))
        nimpa.array2nii(mu, mudct['affine'], fmu, trnsp=mudct['transpose'], flip=mudct['flip'])

        # > mu-map as reference in PET space
        fres = nimpa.resample_spm(
            imfrms[0], fmu, np.eye(4), fimout=algnpth/(cts[i_acq].name.split('.nii')[0]+'_inPET.nii.gz'),
            del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)

        #~~~~~~~~~~~~~~~~~~~~
        # > get rid of any NaNs
        imd = nimpa.getnii(fres, output='all')
        os.remove(fres)
        im = imd['im']
        im[np.isnan(im)] = 0
        nimpa.array2nii(im, imd['affine'], fres, trnsp=imd['transpose'], flip=imd['flip'])
        fres = Path(fres)
        #~~~~~~~~~~~~~~~~~~~~

        # > centre of mass (CoM)
        fcom = fres.parent/(fres.name.split('.nii')[0]+'_CoM-modified.nii')
        nimpa.centre_mass_corr(fres, fout=fcom)
        print('Modified CoM:', fcom)
        fref[i_acq] = fcom
        #--------------------------------------------

        #--------------------------------------------
        # > ALIGNMENT TO CT
        algn_frm[i_acq] = align_frames(imfrms, imtime, fref[i_acq], Cnt,
            reg_tool='spm', spm_com_corr=True, outpath=algnpth)
        #--------------------------------------------
        
        #--------------------------------------------
        # > PET sum of aligned frames
        refdct = nimpa.getnii(fref[i_acq], output='all')
        fsum[i_acq] = qcpth/('PET_algn_sum_{}.nii.gz'.format(i_acq+1))
        nimpa.array2nii(
            np.sum(algn_frm[i_acq]['im4d'],axis=0),
            refdct['affine'],
            fsum[i_acq],
            trnsp=refdct['transpose'], flip=refdct['flip'])
        #--------------------------------------------

    #--------------------------------------------
    # > co-register CTs with the second (as in static imaging) acting as the reference
    ct_reg = nimpa.coreg_spm(fref[1], fref[0], outpath=algnFpth,  costfun='nmi', fwhm_ref=2., fwhm_flo=2.)

    frct = nimpa.resample_spm(fref[1], fref[0], ct_reg['affine'],
        fimout=algnFpth/(fref[0].name.split('.nii')[0]+'_registered-to-CT.nii.gz'),
        del_ref_uncmpr=True, del_flo_uncmpr=True, del_out_uncmpr=True)
    #--------------------------------------------


    #--------------------------------------------
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
        frpet = nimpa.resample_spm(fref[1], f, affF,
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
        #fnii_org.append(niidat['series'][i_acq][keys2[fi]]['fnii'])

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
    # > number of frames for break dynamic acquisitions A and B
    nfrmA = len(niidat['series'][0])
    nfrmB = len(niidat['series'][1])

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



    # > output dictionary
    outdct = {}
    outdct['align_acq'] = algn_frm
    outdct['ctref'] = fref
    outdct['fqcsum'] = fsum
    outdct['ct_reg'] = ct_reg
    outdct['fct_reg'] = frct
    outdct['fur'] = fur
    outdct['fsum'] = fsumF
    outdct['faligned'] = faligned
    outdct['fpet4d'] = f4d
    outdct['faffines'] = faffines
    outdct['fnii_com'] = algn_frm[0]['fnii_com'] + algn_frm[1]['fnii_com']

    return outdct

