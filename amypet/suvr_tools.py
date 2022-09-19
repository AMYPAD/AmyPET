import logging
import os
from pathlib import Path, PurePath
from subprocess import run

import dcm2niix
import numpy as np
from matplotlib import pyplot as plt
from niftypet import nimpa

logging.basicConfig(level=logging.INFO)
nifti_ext = ('.nii', '.nii.gz')
dicom_ext = ('.DCM', '.dcm', '.img', '.IMG', '.ima', '.IMA')


# ========================================================================================
def r_trimup(fpet, fmri, outpath=None, store_img_intrmd=True):
    '''
    trim and upscale PET relative to MR T1w or its derivative;
    derives the scale of upscaling/trimming using the image/voxel sizes
    '''

    if isinstance(fpet, (str, PurePath)):
        petdct = nimpa.getnii(fpet, output='all')
    elif isinstance(fpet, dict) and 'hdr' in fpet:
        petdct = fpet
    else:
        raise ValueError('wrong PET input - accepted are path to image file or dictionary')

    if isinstance(fmri, (str, PurePath)):
        mridct = nimpa.getnii(fmri, output='all')
    elif isinstance(fmri, dict) and 'hdr' in fmri:
        mridct = fmri
    else:
        raise ValueError('wrong MR input - accepted are path to image file or dictionary')

    # > get the voxel sizes
    pet_szyx = petdct['hdr']['pixdim'][1:4]
    mri_szyx = mridct['hdr']['pixdim'][1:4]

    # > estimate the scale
    scale = np.abs(np.round(pet_szyx[::-1] / mri_szyx[::-1])).astype(np.int32)

    # > trim the PET image for more accurate regional sampling
    ftrm = nimpa.imtrimup(fpet, scale=scale, store_img_intrmd=store_img_intrmd, outpath=outpath)

    # > trimmed folder
    trmdir = Path(ftrm['fimi'][0]).parent

    return {'im': ftrm['im'], 'trmdir': trmdir, 'ftrm': ftrm['fimi'][0], 'trim_scale': scale}


# ========================================================================================
def extract_vois(impet, imlabel, voi_dct, outpath=None, output_masks=False):
    '''
    Extract VOI mean values from PET image `impet` using image labels `imlabel`.
    Both can be dictionaries, file paths or Numpy arrays.
    They have to be aligned and have the same dimensions.
    If path (output) is given, the ROI masks will be saved to file(s).
    Arguments:
        - impet:    PET image as Numpy array
        - imlabel:  image of labels (integer values); the labels can come
                    from T1w-based parcellation or an atlas.
        - voi_dct:  dictionary of VOIs, with entries of labels creating
                    composite volumes
        - output_masks: if `True`, output Numpy VOI masks in the output
                    dictionary
        - outpath:  if given as a folder path, the VOI masks will be saved
    '''

    # > assume none of the below are given
    # > used only for saving ROI mask to file if requested
    affine, flip, trnsp = None, None, None

    # ----------------------------------------------
    # PET
    if isinstance(impet, dict):
        im = impet['im']
        if 'affine' in impet:
            affine = impet['affine']
        if 'flip' in impet:
            flip = impet['flip']
        if 'transpose' in impet:
            trnsp = impet['transpose']

    elif isinstance(impet, (str, PurePath)) and os.path.isfile(impet):
        imd = nimpa.getnii(impet, output='all')
        im = imd['im']
        flip = imd['flip']
        trnsp = imd['transpose']

    elif isinstance(impet, np.ndarray):
        im = impet
    # ----------------------------------------------

    # ----------------------------------------------
    # LABELS
    if isinstance(imlabel, dict):
        lbls = imlabel['im']
        if 'affine' in imlabel and affine is None:
            affine = imlabel['affine']
        if 'flip' in imlabel and flip is None:
            flip = imlabel['flip']
        if 'transpose' in imlabel and trnsp is None:
            trnsp = imlabel['transpose']

    elif isinstance(imlabel, (str, PurePath)) and os.path.isfile(imlabel):
        prd = nimpa.getnii(imlabel, output='all')
        lbls = prd['im']
        if affine is None:
            affine = prd['affine']
        if flip is None:
            flip = prd['flip']
        if trnsp is None:
            trnsp = prd['transpose']

    elif isinstance(imlabel, np.ndarray):
        lbls = imlabel

    # > get rid of NaNs if any in the parcellation/label image
    lbls[np.isnan(lbls)] = 0
    # ----------------------------------------------

    # ----------------------------------------------
    # > output dictionary
    out = {}

    logging.debug('Extracting volumes of interest (VOIs):')
    for k, voi in enumerate(voi_dct):

        logging.info(f'  VOI: {voi}')

        # > ROI mask
        rmsk = np.zeros(lbls.shape, dtype=bool)
        # > number of voxels in the ROI
        vxsum = 0
        # > voxel emission sum
        emsum = 0

        for ri in voi_dct[voi]:
            logging.debug(f'   label{ri}')
            rmsk += np.equal(lbls, ri)

        if outpath is not None and not isinstance(imlabel, np.ndarray):
            nimpa.create_dir(outpath)
            fvoi = Path(outpath) / (str(voi) + '_mask.nii.gz')
            nimpa.array2nii(rmsk.astype(np.int8), affine, fvoi,
                            trnsp=(trnsp.index(0), trnsp.index(1), trnsp.index(2)), flip=flip)
        else:
            fvoi = None

        vxsum += np.sum(rmsk)
        emsum += np.sum(im[rmsk].astype(np.float64))

        out[voi] = {'vox_no': vxsum, 'sum': emsum, 'avg': emsum / vxsum, 'fvoi': fvoi}

        if output_masks:
            out[voi]['roimsk'] = rmsk

    # ----------------------------------------------

    return out


# ========================================================================================
def preproc_suvr(pet_path, frames=None, outpath=None, fname=None):
    ''' Prepare the PET image for SUVr analysis.
        Arguments:
        - pet_path: path to the folder of DICOM images, or to the NIfTI file
        - outpath:  output folder path; if not given will assume the parent
                    folder of the input image
        - fname:    core name of the static (SUVr) NIfTI file
        - frames:   list of frames to be used for SUVr processing
    '''

    if not os.path.exists(pet_path):
        raise ValueError('The provided path does not exist')

    # > convert the path to Path object
    pet_path = Path(pet_path)

    # --------------------------------------
    # > sort out the output folder
    if outpath is None:
        petout = pet_path.parent
    else:
        petout = Path(outpath)

    nimpa.create_dir(petout)

    if fname is None:
        fname = nimpa.rem_chars(pet_path.name.split('.')[0]) + '_static.nii.gz'
    elif not str(fname).endswith(nifti_ext[1]):
        fname += '.nii.gz'
    # --------------------------------------

    # > NIfTI case
    if pet_path.is_file() and str(pet_path).endswith(nifti_ext):
        logging.info('PET path exists and it is a NIfTI file')

        fpet_nii = pet_path

    # > DICOM case (if any file inside the folder is DICOM)
    elif pet_path.is_dir() and any([f.suffix in dicom_ext for f in pet_path.glob('*')]):

        # > get the NIfTi images from previous processing
        fpet_nii = list(petout.glob(pet_path.name + '*.nii*'))

        if not fpet_nii:
            run([dcm2niix.bin, '-i', 'y', '-v', 'n', '-o', petout, 'f', '%f_%s', pet_path])

        fpet_nii = list(petout.glob(pet_path.name + '*.nii*'))

        # > if cannot find a file it might be due to spaces in folder/file names
        if not fpet_nii:
            fpet_nii = list(petout.glob(pet_path.name.replace(' ', '_') + '*.nii*'))

        if not fpet_nii:
            raise ValueError('No SUVr NIfTI files found')
        elif len(fpet_nii) > 1:
            raise ValueError('Too many SUVr NIfTI files found')
        else:
            fpet_nii = fpet_nii[0]

    # > read the dynamic image
    imdct = nimpa.getnii(fpet_nii, output='all')

    # > number of dynamic frames
    nfrm = imdct['hdr']['dim'][4]

    # > ensure that the frames exist in part of full dynamic image data
    if frames and nfrm < max(frames):
        raise ValueError('The selected frames do not exist')
    elif not frames:
        frames = np.arange(nfrm)

    logging.info(f'{nfrm} frames have been found in the dynamic image.')

    # ------------------------------------------
    # > static image file path
    fstat = petout / fname

    # > check if the static (for SUVr) file already exists
    if not fstat.is_file():

        if nfrm > 1:
            imstat = np.sum(imdct['im'][frames, ...], axis=0)
        else:
            imstat = np.squeeze(imdct['im'])

        nimpa.array2nii(
            imstat, imdct['affine'], fstat,
            trnsp=(imdct['transpose'].index(0), imdct['transpose'].index(1),
                   imdct['transpose'].index(2)), flip=imdct['flip'])

        logging.info(f'Saved SUVr file image to: {fstat}')
    # ------------------------------------------

    return {'fpet_nii': fpet_nii, 'fstat': fstat}


# ========================================================================================
# Extract VOI values for SUVr analysis (main function)
# ========================================================================================


def voi_process(petpth, lblpth, t1wpth, voi_dct=None, ref_voi=None, frames=None, fname=None,
                t1_bias_corr=True, outpath=None, output_masks=True, save_voi_masks=False,
                qc_plot=True, reg_fwhm_pet=0, reg_fwhm_mri=0, reg_costfun='nmi', reg_fresh=True):
    ''' Process PET image for VOI extraction using MR-based parcellations.
        The T1w image and the labels which are based on the image must be
        in the same image space.

        Arguments:
        - petpth:   path to the PET NIfTI image
        - lblpth:   path to the label NIfTI image (parcellations)
        - t1wpth:   path to the T1w MRI NIfTI image for registration
        - voi_dct:  dictionary of VOI definitions
        - ref_voi:  if given and in `voi_dct` it is used as reference region
                    for calculating SUVr
        - frames:   select the frames if multi-frame image given;
                    by default selects all frames
        - fname:    the core file name for resulting images
        - t1_bias_corr: it True, performs bias field correction of the T1w image
        - outpath:  folder path to the output images, including intermediate
                    images
        - output_masks: if True, output VOI sampling masks in the output
                    dictionary
        - save_voi_masks: if True, saves all the VOI masks to the `masks` folder
        - qc_plot:  plots the PET images and overlay sampling, and saves it to
                    a PNG file; requires `output_masks` to be True.
        - reg_fwhm: FWHMs of the Gaussian filter applied to PET or MRI images
                    by default 0 mm;
        - reg_costfun: cost function used in image registration
        - reg_fresh:runs fresh registration if True, otherwise uses an existing
                    one if found.

    '''

    # > output dictionary
    out = {}

    # > make sure the paths are Path objects
    petpth = Path(petpth)
    t1wpth = Path(t1wpth)
    lblpth = Path(lblpth)

    if outpath is None:
        outpath = petpth.parent
    else:
        outpath = Path(outpath)

    out['input'] = {'fpet': petpth, 'ft1w': t1wpth, 'flbl': lblpth}

    if not (petpth.exists() and t1wpth.is_file() and lblpth.is_file()):
        raise ValueError('One of the three paths to PET, T1w or label image is incorrect.')

    # > if dictionary is not given, the VOI values will be calculated for each unique
    # > VOI in the label/parcellation image
    if voi_dct is None:
        lbl = nimpa.getnii(lblpth)
        voi_dct = {int(lab): [int(lab)] for lab in np.unique(lbl)}

    if ref_voi is not None and not all([r in voi_dct for r in ref_voi]):
        raise ValueError('Not all VOIs listed as reference are in the VOI definition dictionary.')

    # > static (SUVr) image preprocessing
    suvr_preproc = preproc_suvr(petpth, frames=frames,
                                outpath=outpath / (petpth.name.split('.')[0] + '_suvr'),
                                fname=fname)

    out.update(suvr_preproc)

    if t1_bias_corr:
        out['n4'] = nimpa.bias_field_correction(t1wpth, executable='sitk',
                                                outpath=suvr_preproc['fstat'].parent.parent)
        fmri = out['n4']['fim']
    else:
        fmri = t1wpth

    # --------------------------------------------------
    # TRIMMING / UPSCALING
    # > derive the scale of upscaling/trimming using the current
    # > image/voxel sizes
    trmout = r_trimup(suvr_preproc['fstat'], lblpth, store_img_intrmd=True)

    # > trimmed folder
    trmdir = trmout['trmdir']

    # > trimmed and upsampled PET file
    out['ftrm'] = trmout['ftrm']
    out['trim_scale'] = trmout['trim_scale']
    # --------------------------------------------------

    # > - - - - - - - - - - - - - - - - - - - - - - - -
    # > parcellations in PET space
    fplbl = trmdir / '{}_Parcellation_in-upsampled-PET.nii.gz'.format(
        suvr_preproc['fstat'].name.split('.nii')[0])

    if not fplbl.is_file() or reg_fresh:

        logging.info(f'registration with smoothing of {reg_fwhm_pet}, {reg_fwhm_mri} mm'
                     ' for reference and floating images respectively')

        spm_res = nimpa.coreg_spm(trmout['ftrm'], fmri, fwhm_ref=reg_fwhm_pet,
                                  fwhm_flo=reg_fwhm_mri, fwhm=[7, 7], costfun=reg_costfun,
                                  fcomment='', outpath=trmdir, visual=0, save_arr=False,
                                  del_uncmpr=True)

        flbl_pet = nimpa.resample_spm(
            trmout['ftrm'],
            lblpth,
            spm_res['faff'],
            outpath=trmdir,
            intrp=0.,
            fimout=fplbl,
            del_ref_uncmpr=True,
            del_flo_uncmpr=True,
            del_out_uncmpr=True,
        )

    out['flbl'] = fplbl
    # > - - - - - - - - - - - - - - - - - - - - - - - -

    # > get the label image in PET space
    plbl_dct = nimpa.getnii(fplbl, output='all')

    # > get the sampling output
    if save_voi_masks:
        mask_dir = trmdir / 'masks'
    else:
        mask_dir = None
    voival = extract_vois(trmout['im'], plbl_dct, voi_dct, outpath=mask_dir,
                          output_masks=output_masks)

    # > calculate SUVr if reference regions is given
    suvrtxt = None
    if ref_voi is not None:

        suvr = {}

        suvrtxt = ' '
        for rvoi in ref_voi:
            ref = voival[rvoi]['avg']
            suvr[rvoi] = {}
            for voi in voi_dct:
                suvr[rvoi][voi] = voival[voi]['avg'] / ref

            # > get the static trimmed image:
            imsuvr = nimpa.getnii(out['ftrm'], output='all')

            fsuvr = trmdir / 'SUVr_ref-{}_{}'.format(rvoi, suvr_preproc['fstat'].name)
            # > save SUVr image
            nimpa.array2nii(
                imsuvr['im'] / ref, imsuvr['affine'], fsuvr,
                trnsp=(imsuvr['transpose'].index(0), imsuvr['transpose'].index(1),
                       imsuvr['transpose'].index(2)), flip=imsuvr['flip'])

            suvr[rvoi]['fsuvr'] = fsuvr

            if 'suvr' in voi_dct:
                suvrval = suvr[rvoi]['suvr']
                suvrtxt += f'$SUVR_\\mathrm{{{rvoi}}}=${suvrval:.3f}; '

        out['suvr'] = suvr

    out['vois'] = voival

    # -----------------------------------------
    # > QC plot
    if qc_plot and output_masks:
        showpet = nimpa.imsmooth(trmout['im'].astype(np.float32), voxsize=plbl_dct['voxsize'],
                                 fwhm=3.)

        def axrange(prf, thrshld, parts):
            zs = next(x for x, val in enumerate(prf) if val > thrshld)
            ze = len(prf) - next(x for x, val in enumerate(prf[::-1]) if val > thrshld)
            # divide the range in parts
            p = int((ze-zs) / parts)
            zn = []
            for k in range(1, parts):
                zn.append(zs + k*p)
            return zn

        # z-profile
        zn = []
        thrshld = 100
        zprf = np.sum(voival['neocx']['roimsk'], axis=(1, 2))
        zn += axrange(zprf, thrshld, 3)

        zprf = np.sum(voival['cblgm']['roimsk'], axis=(1, 2))
        zn += axrange(zprf, thrshld, 2)

        mskshow = voival['neocx']['roimsk'] + voival['cblgm']['roimsk']

        xn = []
        xprf = np.sum(mskshow, axis=(0, 1))
        xn += axrange(xprf, thrshld, 4)

        fig, ax = plt.subplots(2, 3, figsize=(16, 16))

        for ai, zidx in enumerate(zn):
            msk = mskshow[zidx, ...]
            impet = showpet[zidx, ...]
            ax[0][ai].imshow(impet, cmap='magma', vmax=0.9 * impet.max())
            ax[0][ai].imshow(msk, cmap='gray_r', alpha=0.25)
            ax[0][ai].xaxis.set_visible(False)
            ax[0][ai].yaxis.set_visible(False)

        for ai, xidx in enumerate(xn):
            msk = mskshow[..., xidx]
            impet = showpet[..., xidx]
            ax[1][ai].imshow(impet, cmap='magma', vmax=0.9 * impet.max())
            ax[1][ai].imshow(msk, cmap='gray_r', alpha=0.25)
            ax[1][ai].xaxis.set_visible(False)
            ax[1][ai].yaxis.set_visible(False)

        ax[0, 1].text(0, trmout['im'].shape[1] + 10, suvrtxt, fontsize=12)

        plt.tight_layout()

        fqc = trmdir / f'QC_{petpth.name}_Parcellation-over-upsampled-PET.png'
        plt.savefig(fqc, dpi=300)
        plt.close('all')
        out['fqc'] = fqc
    # -----------------------------------------

    return out
