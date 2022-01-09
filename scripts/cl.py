"""Run centiloid pipeline"""
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spm12
from miutil.fdio import hasext
from niftypet import nimpa

log = logging.getLogger(__name__)


def run(fpets, fmris, atlases, outpath=None, visual=False):
    """
    Process centiloid (CL) using input file lists for PET and MRI
    images, <fpets> and <fmri>.  The images must be in NIfTI format.
    Args:
      atlases: the path to the CL 2mm resolution atlases
      outpath: path to the output folder
    """
    spm_path = Path(spm12.utils.spm_dir())
    atlases = Path(atlases)
    out = {}                                           # output dictionary
    tmpl_avg = spm_path / 'canonical' / 'avg152T1.nii' # template path

    if isinstance(fpets, (str, Path)) and isinstance(fmris, (str, Path)):
        # when single PET and MR files are given
        if os.path.isfile(fpets) and os.path.isfile(fmris):
            pet_mr_list = [[fpets], [fmris]]
        # when folder paths are given for PET and MRI files
        # the content is sorted for both PET and MRI and then matched
        # according to the order
        elif os.path.isdir(fpets) and os.path.isdir(fmris):
            fp = sorted(
                os.path.join(fpets, f) for f in os.listdir(fpets) if hasext(f, ('nii', 'nii.gz')))
            fm = sorted(
                os.path.join(fmris, f) for f in os.listdir(fmris) if hasext(f, ('nii', 'nii.gz')))
            pet_mr_list = [fp, fm]
        else:
            raise ValueError('unrecognised or unmatched input for PET and MRI image data')
    elif isinstance(fpets, list) and isinstance(fmris, list):
        if len(fpets) != len(fmris):
            raise ValueError('the number of PET and MRI files have to match!')
        # check if all files exists
        if not all(os.path.isfile(f) and hasext(f, ('nii', 'nii.gz')) for f in fpets) or not all(
                os.path.isfile(f) and hasext(f, ('nii', 'nii.gz')) for f in fmris):
            raise ValueError('the number of paired files do not match')
        pet_mr_list = [fpets, fmris]
    else:
        raise ValueError('unrecognised input image data')

    log.info('loading CL masks...')
    fmasks = {
        'crbl_gm': atlases / 'voi_CerebGry_2mm.nii', 'crbl': atlases / 'voi_WhlCbl_2mm.nii',
        'crbl_bs': atlases / 'voi_WhlCblBrnStm_2mm.nii', 'pons': atlases / 'voi_Pons_2mm.nii',
        'crtx': atlases / 'voi_ctx_2mm.nii'}
    masks = {fmsk: nimpa.getnii(fmasks[fmsk]) for fmsk in fmasks}

    log.info('iterate through all the input data...')
    for fpet, fmri in zip(*pet_mr_list):
        # make the files Path objects
        fpet, fmri = Path(fpet), Path(fmri)
        opth = Path(fpet.parent if outpath is None else outpath)

        # create output dictionary and folder specific to the PET
        # file which will be CL quantified; name of the folder
        # and dictionary output is based on PET name
        onm = fpet.name.rsplit('.nii', 1)[0]
        spth = opth / onm
        out[onm] = {'opth': spth, 'fpet': fpet, 'fmri': fmri}
        # output path for centre of mass alignment and registration
        opthc = spth / 'centre-of-mass'
        opthr = spth / 'registration'
        opthn = spth / 'normalisation'
        optho = spth / 'normalised'
        opths = spth / 'suvr'

        log.info(f'subject {onm}: centre of mass correction')
        # modify for the centre of mass being at O(0,0,0)
        out[onm]['petc'] = petc = nimpa.centre_mass_corr(fpet, outpath=opthc)
        out[onm]['mric'] = mric = nimpa.centre_mass_corr(fmri, outpath=opthc)

        log.info(f'subject {onm}: MR registration to MNI space')
        out[onm]['reg1'] = reg1 = spm12.coreg_spm(
            tmpl_avg,
            mric['fim'],
            fwhm_ref=0,
            fwhm_flo=3,
            outpath=opthr,
            fname_aff="",
            fcomment="",
            pickname="ref",
            costfun="nmi",
            graphics=1,
            visual=int(visual),
            del_uncmpr=True,
            save_arr=True,
            save_txt=True,
            modify_nii=True,
        )

        log.info(f'subject {onm}: PET -> MR registration')
        out[onm]['reg2'] = reg2 = spm12.coreg_spm(
            reg1['freg'],
            petc['fim'],
            fwhm_ref=3,
            fwhm_flo=6,
            outpath=opthr,
            fname_aff="",
            fcomment='_mr-reg',
            pickname="ref",
            costfun="nmi",
            graphics=1,
            visual=int(visual),
            del_uncmpr=True,
            save_arr=True,
            save_txt=True,
            modify_nii=True,
        )

        log.info(f'subject {onm}:MR normalisation/segmentation...')
        out[onm]['norm'] = norm = spm12.seg_spm(reg1['freg'], spm_path, outpath=opthn,
                                                store_nat_gm=False, store_nat_wm=False,
                                                store_nat_csf=True, store_fwd=True, store_inv=True,
                                                visual=visual)
        # normalise
        list4norm = [reg1['freg'] + ',1', reg2['freg'] + ',1']
        out[onm]['fnorm'] = spm12.normw_spm(norm['fordef'], list4norm, outpath=optho)

        log.info(f'subject {onm}: load normalised PET image...')
        fnpets = [f for f in optho.iterdir() if fpet.name.split('.nii')[0] in f.name]

        if len(fnpets) == 0:
            raise ValueError('could not find normalised PET image files')
        elif len(fnpets) > 1:
            raise ValueError('too many potential normalised PET image files found')

        npet = nimpa.getnii(fnpets[0])
        npet[np.isnan(npet)] = 0 # get rid of NaNs

        # npet[npet<0] = 0

        # extract mean values and SUVr
        out[onm]['avgvoi'] = avgvoi = {fmsk: np.mean(npet[masks[fmsk] > 0]) for fmsk in fmasks}
        out[onm]['suvr'] = suvr = {
            fmsk: avgvoi['crtx'] / avgvoi[fmsk]
            for fmsk in fmasks if fmsk != 'crtx'}

        # --------------------------------------------------------------
        # VISUALISATION
        # pick masks for visualisation
        msk = 'crtx'
        mskr = 'crbl' # 'pons'#

        nimpa.create_dir(opths)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # transaxial tiling for QC
        # choose the shape of the mosaic/tiled image
        izs = np.array([[30, 40, 50], [60, 70, 80]])

        # get the shape of the mosaic
        shp = izs.shape

        # initialise mosaic for PET and masks
        mscp_t = np.zeros((shp[0], shp[1]) + npet[0, ...].shape, dtype=np.float32)
        mscm_t = mscp_t.copy()

        # fill in the images
        for i in range(shp[0]):
            for j in range(shp[1]):
                mscp_t[i, j, ...] = npet[izs[i, j], ...]
                mscm_t[i, j, ...] = masks[msk][izs[i, j], ...] + masks[mskr][izs[i, j], ...]

        # reshape for the final touch
        mscp_t = mscp_t.swapaxes(1, 2)
        mscm_t = mscm_t.swapaxes(1, 2)
        mscp_t = mscp_t.reshape(shp[0] * npet.shape[1], shp[1] * npet.shape[2])
        mscm_t = mscm_t.reshape(shp[0] * npet.shape[1], shp[1] * npet.shape[2])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Sagittal tiling for QC

        # choose the shape of the mosaic/tiled image
        ixs = np.array([[20, 30, 40], [50, 60, 70]])

        # get the shape of the mosaic
        shp = ixs.shape

        # initialise mosaic for PET and masks
        mscp_s = np.zeros((shp[0], shp[1]) + npet[..., 0].shape, dtype=np.float32)
        mscm_s = mscp_s.copy()

        # fill in the images
        for i in range(shp[0]):
            for j in range(shp[1]):
                mscp_s[i, j, ...] = npet[..., ixs[i, j]]
                mscm_s[i, j, ...] = masks[msk][..., ixs[i, j]] + masks[mskr][..., ixs[i, j]]

        # reshape for the final touch
        mscp_s = mscp_s.swapaxes(1, 2)
        mscm_s = mscm_s.swapaxes(1, 2)
        mscp_s = mscp_s.reshape(shp[0] * npet.shape[0], shp[1] * npet.shape[1])
        mscm_s = mscm_s.reshape(shp[0] * npet.shape[0], shp[1] * npet.shape[1])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        fig, ax = plt.subplots(2, 1, figsize=(9, 12))

        thrsh = 0.6 * npet.max()

        ax[0].imshow(mscp_t, cmap='magma', vmax=thrsh)
        ax[0].imshow(mscm_t, cmap='gray_r', alpha=0.25)
        ax[0].set_axis_off()
        ax[0].set_title(f'{onm}: transaxial centiloid sampling')

        ax[1].imshow(mscp_s, cmap='magma', vmax=thrsh)
        ax[1].imshow(mscm_s, cmap='gray_r', alpha=0.25)
        ax[1].set_axis_off()
        ax[1].set_title(f'{onm} sagittal centiloid sampling')

        suvrstr = ",   ".join([
            f"$SUVR_{{WC}}=${suvr['crbl']:.3f}", f"$SUVR_{{GMC}}=${suvr['crbl_gm']:.3f}",
            f"$SUVR_{{CBS}}=${suvr['crbl_bs']:.3f}", f"$SUVR_{{PNS}}=${suvr['pons']:.3f)}"])

        ax[1].text(0, 200, suvrstr, fontsize=12)
        plt.tight_layout()

        fqcpng = opths / 'CL_mask_PET_sampling.png'
        plt.savefig(fqcpng, dpi=150, facecolor='auto', edgecolor='auto')

        plt.close('all')

        out[onm]['fqc'] = fqcpng
    return out