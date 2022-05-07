"""Centiloid pipeline

Usage:
  centiloid [options] <fpets> <fmris> <atlases>

Arguments:
  <fpets>  : PET NIfTI directory [default: DirChooser]
  <fmris>  : MRI NIfTI directory [default: DirChooser]
  <atlases>  : Reference regions ROIs directory
    (standard Centiloid RR from GAAIN Centioid website: 2mm, NIfTI)
    [default: DirChooser]

Options:
  --outpath FILE  : Output directory
  --visual  : whether to plot
"""

__author__ = "Pawel J Markiewicz, Casper Da Costa-Luis"
__copyright__ = "Copyright 2022"

import logging
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import spm12
from miutil.fdio import hasext, nsort
from niftypet import nimpa

log = logging.getLogger(__name__)


#----------------------------------------------------------------------
def load_masks(atlases):
    ''' Load the Centiloid PET masks for calculating 
        the SUVr to then convert it to Centiloid.

        Return the paths to the masks and the masks themselves
    '''

    log.info('loading CL masks...')
    fmasks = {
        'cg': atlases / 'voi_CerebGry_2mm.nii', 'wc': atlases / 'voi_WhlCbl_2mm.nii',
        'wcb': atlases / 'voi_WhlCblBrnStm_2mm.nii', 'pns': atlases / 'voi_Pons_2mm.nii',
        'ctx': atlases / 'voi_ctx_2mm.nii'}
    masks = {fmsk: nimpa.getnii(fmasks[fmsk]) for fmsk in fmasks}

    return fmasks, masks
#----------------------------------------------------------------------



def run(fpets,
        fmris,
        atlases,
        tracer='pib',
        flip_pet=None,
        bias_corr=True,
        outpath=None,
        visual=False,
        climage=True,
        used_saved=False,
        cl_anchore_path=None):
    """
    Process centiloid (CL) using input file lists for PET and MRI
    images, `fpets` and `fmris` (must be in NIfTI format).
    Args:
      atlases: the path to the CL 2mm resolution atlases
      outpath: path to the output folder
      bias_corr: bias filed correction for the MR image (True/False)
      tracer: specifies what tracer is being used and so the right
              transformation is used; by default it is PiB.  Currently
              [18F]flutemetamol, 'flute', and [18F]florbetaben, 'fbb'
              are supported. 
              IMPORTANT: when calibrating a new tracer, ensure that
              `tracer`='new'.
      flip_pet: a list of flips (3D tuples) which flip any dimension
               of the 3D PET image (z,y,x); the list has to have the 
               same length as the lists of `fpets` and `fmris` 
      used_saved: if True, looks for already saved normalised PET
                images and loads them to avoid processing time.
      visual: SPM-based progress visualisations of image registration or
              or image normalisation.
      climage: outputs the CL-converted PET image in the MNI space
      cl_anchore_path: The path where optional CL anchor dictionary is
                saved.
    """


    # supported F-18 tracers
    f18_tracers =  ['fbp', 'fbb', 'flute']


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
            fp = nsort(
                os.path.join(fpets, f) for f in os.listdir(fpets) if hasext(f, ('nii', 'nii.gz')))
            fm = nsort(
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


    #-------------------------------------------------------------
    if flip_pet is not None:
        if isinstance(flip_pet, tuple) and len(pet_mr_list[0])==1:
            flips = [flip_pet]
        elif isinstance(flip_pet, list) and len(flip_pet)==len(pet_mr_list[0]):
            flips = flip_pet
        else:
            log.warning('the flip definition is not compatible with the list of PET images')
    else:
        flips = [None]*len(pet_mr_list[0])
    #-------------------------------------------------------------


    fmasks, masks = load_masks(atlases)


    log.info('iterate through all the input data...')
    
    fi = -1
    for fpet, fmri in zip(*pet_mr_list):

        fi+=1
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
        opthi = spth / 'cl-image'
 
        # find if the normalised PET is already there
        if optho.is_dir():
            fnpets = [f for f in optho.iterdir() if fpet.name.split('.nii')[0] in f.name]
        else:
            fnpets = []
        
        if used_saved and len(fnpets)==1:
            log.info(f'subject {onm}: loading already normalised PET image...')

        else:

            # run bias field correction unless cancelled
            if bias_corr:
                log.info(f'subject {onm}: running MR bias field correction')
                out[onm]['n4'] = nimpa.bias_field_correction(
                    fmri,
                    executable = 'sitk',
                    outpath = spth)
                fmri = out[onm]['n4']['fim']


            log.info(f'subject {onm}: centre of mass correction')
            # check if flipping the PET is requested
            if flips[fi] is not None and any(flips[fi]):
                flip=flips[fi]
            else:
                flip = None

            # modify for the centre of mass being at O(0,0,0)
            out[onm]['petc'] = petc = nimpa.centre_mass_corr(fpet, flip=flip, outpath=opthc)
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

            log.info(f'subject {onm}: MR normalisation/segmentation...')
            out[onm]['norm'] = norm = spm12.seg_spm(reg1['freg'], spm_path, outpath=opthn,
                                                    store_nat_gm=False, store_nat_wm=False,
                                                    store_nat_csf=True, store_fwd=True, store_inv=True,
                                                    visual=visual)
            # normalise
            list4norm = [reg1['freg'] + ',1', reg2['freg'] + ',1']
            out[onm]['fnorm'] = spm12.normw_spm(norm['fordef'], list4norm, outpath=optho)

            log.info(f'subject {onm}: load normalised PET image...')
            fnpets = [f for f in optho.iterdir() \
                        if fpet.name.split('.nii')[0] in f.name \
                        and not 'n4bias' in f.name.lower()]# and not 'mr' in f.name.lower()]

            if len(fnpets) == 0:
                raise ValueError('could not find normalised PET image files')
            elif len(fnpets) > 1:
                raise ValueError('too many potential normalised PET image files found')

        npet_dct = nimpa.getnii(fnpets[0], output='all')
        npet = npet_dct['im']
        npet[np.isnan(npet)] = 0 # get rid of NaNs

        # npet[npet<0] = 0

        # extract mean values and SUVr
        out[onm]['avgvoi'] = avgvoi = {fmsk: np.mean(npet[masks[fmsk] > 0]) for fmsk in fmasks}
        out[onm]['suvr'] = suvr = {
            fmsk: avgvoi['ctx'] / avgvoi[fmsk]
            for fmsk in fmasks if fmsk != 'ctx'}

        #**************************************************************
        # C E N T I L O I D   S C A L I N G
        #**************************************************************
        #---------------------------------
        # > path to anchor point dictionary
        if cl_anchore_path is None:
            cpth = os.path.realpath(__file__)
        elif os.path.exists(cl_anchore_path):
            cpth = Path(cl_anchore_path)
        #---------------------------------


        #---------------------------------
        # > centiloid transformation for PiB
        pth = os.path.join(os.path.dirname(cpth), 'CL_PiB_anchors.pkl')

        if not os.path.isfile(pth):
            tracer = 'new'
            log.warning('Could not find the PiB CL anchor point definitions/tables')

        with open(pth, 'rb') as f:
            CLA = pickle.load(f)
        #---------------------------------


        #---------------------------------
        # > centiloid transformation for PiB
        # > check if SUVr transformation is needed for F-18 tracers
        if tracer in f18_tracers:
            pth = os.path.join(os.path.dirname(cpth), f'suvr_{tracer}_to_suvr_pib__transform.pkl')

            if not os.path.isfile(pth):
                log.warning('The conversion dictionary/table for the specified tracer is not found')
            else:
                with open(pth, 'rb') as f:
                    CNV = pickle.load(f)
                print('loaded SUVr transformations for tracer:', tracer)

                # > save the PiB converted SUVrs
                out[onm]['suvr_pib_calc'] = suvr_pib_calc = {
                    fmsk: (suvr[fmsk] - CNV[fmsk]['b_std']) / CNV[fmsk]['m_std']
                    for fmsk in fmasks if fmsk != 'ctx'}

                # > save the linear transformation parameters 
                out[onm]['suvr_pib_calc_transf'] = {
                    fmsk: (CNV[fmsk]['m_std'], CNV[fmsk]['b_std'])
                    for fmsk in fmasks if fmsk != 'ctx'}
                    
            # > used now the new PiB converted SUVrs
            suvr = suvr_pib_calc
        #---------------------------------
        

        if tracer!='new':
            out[onm]['cl'] = cl = {
                fmsk: 100*(suvr[fmsk]-CLA[fmsk][0])/(CLA[fmsk][1]-CLA[fmsk][0])
                for fmsk in fmasks if fmsk != 'ctx'}
        
        #**************************************************************

        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # > output the CL-converted PET image in the MNI space
        if climage and tracer!='new':

            for refvoi in fmasks:
                if refvoi=='ctx': continue

                refavg = out[onm]['avgvoi'][refvoi]

                # > obtain the SUVr image for the given reference VOI
                npet_suvr = npet/refavg
            
                # > convert to PiB scale if it is an F18 tracer
                if tracer in f18_tracers:
                    npet_suvr = (npet_suvr - CNV[refvoi]['b_std']) / CNV[refvoi]['m_std']

                # > convert the (PiB) SUVr image to CL scale
                npet_cl = 100*(npet_suvr-CLA[refvoi][0])/(CLA[refvoi][1]-CLA[refvoi][0])

                # > get the CL global value by applying the CTX mask
                cl_ = np.mean(npet_cl[masks['ctx'].astype(bool)])


                cl_refvoi = cl[refvoi]
                if tracer!='new' and abs(cl_-cl_refvoi)<0.25:
                    # > save the CL-converted file
                    nimpa.create_dir(opthi)
                    fout = opthi / ('CL_image_ref-'+refvoi+fnpets[0].name.split('.nii')[0]+'.nii.gz')
                    nimpa.array2nii(
                        npet_cl,
                        npet_dct['affine'],
                        fout,
                        trnsp = (npet_dct['transpose'].index(0),
                                 npet_dct['transpose'].index(1),
                                 npet_dct['transpose'].index(2)),
                        flip = npet_dct['flip'])

                elif tracer!='new' and abs(cl_-cl_refvoi)>0.25:
                    log.warning(f'The CL of CL-converted image is different to the calculated CL (CL_img={cl_:.4f} vs CL={cl_refvoi:.4f}).')
                    log.warning(f'The CL image has not been generated!')
                else:
                    log.warning(f'The CL image has not been generated due to new tracer being used')
        #-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


        # -------------------------------------------------------------
        # VISUALISATION
        # pick masks for visualisation
        msk = 'ctx'
        mskr = 'wc' # 'pons'#

        showpet = nimpa.imsmooth(npet.astype(np.float32), voxsize=npet_dct['voxsize'], fwhm=3.)

        nimpa.create_dir(opths)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # transaxial tiling for QC
        # choose the shape of the mosaic/tiled image
        izs = np.array([[30, 40, 50], [60, 70, 80]])

        # get the shape of the mosaic
        shp = izs.shape

        # initialise mosaic for PET and masks
        mscp_t = np.zeros((shp[0], shp[1]) + showpet[0, ...].shape, dtype=np.float32)
        mscm_t = mscp_t.copy()

        # fill in the images
        for i in range(shp[0]):
            for j in range(shp[1]):
                mscp_t[i, j, ...] = showpet[izs[i, j], ...]
                mscm_t[i, j, ...] = masks[msk][izs[i, j], ...] + masks[mskr][izs[i, j], ...]

        # reshape for the final touch
        mscp_t = mscp_t.swapaxes(1, 2)
        mscm_t = mscm_t.swapaxes(1, 2)
        mscp_t = mscp_t.reshape(shp[0] * showpet.shape[1], shp[1] * showpet.shape[2])
        mscm_t = mscm_t.reshape(shp[0] * showpet.shape[1], shp[1] * showpet.shape[2])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Sagittal tiling for QC

        # choose the shape of the mosaic/tiled image
        ixs = np.array([[20, 30, 40], [50, 60, 70]])

        # get the shape of the mosaic
        shp = ixs.shape

        # initialise mosaic for PET and masks
        mscp_s = np.zeros((shp[0], shp[1]) + showpet[..., 0].shape, dtype=np.float32)
        mscm_s = mscp_s.copy()

        # fill in the images
        for i in range(shp[0]):
            for j in range(shp[1]):
                mscp_s[i, j, ...] = showpet[..., ixs[i, j]]
                mscm_s[i, j, ...] = masks[msk][..., ixs[i, j]] + masks[mskr][..., ixs[i, j]]

        # reshape for the final touch
        mscp_s = mscp_s.swapaxes(1, 2)
        mscm_s = mscm_s.swapaxes(1, 2)
        mscp_s = mscp_s.reshape(shp[0] * showpet.shape[0], shp[1] * showpet.shape[1])
        mscm_s = mscm_s.reshape(shp[0] * showpet.shape[0], shp[1] * showpet.shape[1])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        fig, ax = plt.subplots(2, 1, figsize=(9, 12))

        thrsh = 0.9*showpet.max()

        ax[0].imshow(mscp_t, cmap='magma', vmax=thrsh)
        ax[0].imshow(mscm_t, cmap='gray_r', alpha=0.25)
        ax[0].set_axis_off()
        ax[0].set_title(f'{onm}: transaxial centiloid sampling')

        ax[1].imshow(mscp_s, cmap='magma', vmax=thrsh)
        ax[1].imshow(mscm_s, cmap='gray_r', alpha=0.25)
        ax[1].set_axis_off()
        ax[1].set_title(f'{onm} sagittal centiloid sampling')

        suvrstr = ",   ".join([
            f"PiB transformed: $SUVR_{{WC}}=${suvr['wc']:.3f}", f"$SUVR_{{GMC}}=${suvr['cg']:.3f}",
            f"$SUVR_{{CBS}}=${suvr['wcb']:.3f}", f"$SUVR_{{PNS}}=${suvr['pns']:.3f}"])
        ax[1].text(0, 190, suvrstr, fontsize=12)

        if tracer!='new':
            clstr = ",   ".join([
                f"$CL_{{WC}}=${cl['wc']:.1f}", f"$CL_{{GMC}}=${cl['cg']:.1f}",
                f"$CL_{{CBS}}=${cl['wcb']:.1f}", f"$CL_{{PNS}}=${cl['pns']:.1f}"])
            ax[1].text(0, 205, clstr, fontsize=12)

        plt.tight_layout()

        fqcpng = opths / (onm+'_CL-SUVr_mask_PET_sampling.png')
        plt.savefig(fqcpng, dpi=150, facecolor='auto', edgecolor='auto')

        plt.close('all')

        out[onm]['fqc'] = fqcpng




    return out
