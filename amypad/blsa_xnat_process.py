'''
AMYPAD image data processing.
Download image data from XNAT, process it and upload back
'''

__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2020"
import csv
import glob
import logging
import os
import re
import sys

from niftypet import nimpa
from niftypet import nixnat
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


log = logging.getLogger(__name__)

fwhm_mri = 0.
fwhm_pet = 4.


dcm_ext = nixnat.xnat.dcm_ext

#> run it every time when access to XNAT is needed using credentials
#> fcrdntls which are stored when running setup for the first time.
#: nixnat.setup_access(outpath='', fcrdntls='xnat_blsa.json'):

xc = nixnat.establish_connection(fcrdntls='xnat_blsa.json')

#> path for processing output which will be then uploaded to XNAT
opth = '/home/pawel/AMYPAD/BLSA/xnat-download_' + nixnat.time_stamp(simple_ascii=True)


#===============================================================================
#> explore the XNAT project
#---------------------------------------
#> get list of subjects in json format
sjson = nixnat.get_list(
    xc['sbj'] + '?format=json',
    cookie=xc['cookie'])
#---------------------------------------

#---------------------------------------
#> Go through all available subjects
#> and get the labels and IDs
lbls = [str(s['label']) for s in sjson]
ids  = [str(s['ID']) for s in sjson]
#---------------------------------------
#===============================================================================



#===============================================================================
def trim(fim, fcomment=''):
    '''
    trim and upsample PET images to prepare them for ROI signal extraction
    '''

    imdct = nimpa.getnii(fim, output='all')
    spth = os.path.dirname(fim)

    #------------------------------------
    #> trim and upscale the image
    ptrm = nimpa.trimim( fim,
            affine=None,
            scale=np.int8(imdct['voxsize']),
            divdim = 32, #8**2,
            fmax = 0.05,
            int_order=1,
            outpath=spth,
            fname='',
            fcomment=fcomment,
            store_img=True,
            store_avg=True,
            imdtype=np.float32,
            memlim=False,
            verbose=True,
            Cnt=None)
    #------------------------------------

    return ptrm
#===============================================================================


#===============================================================================
def icom_correction(imdct, Cnt=None, com=None):
    ''' Image centre of mass correction.  The O point is in the middle of the
        image centre of radio-activity mass

        imdct -    must be a dictionary of the input image as by
                    nimpa.getnii(path_im, output='all').  Preferably trimmed using
                    nimpa.trimim().
    '''
    #> check if the dictionary of constants is given
    if Cnt is None:
        Cnt = {}

    #> output the centre of mass if image radiodistribution in each dimension in mm.
    if com is None:
        com = nimpa.centre_mass_img(imdct, output='mm')

    com = np.array(com)

    if not isinstance(com, np.ndarray):
        raise ValueError('The Centre of Mass is not a Numpy array!')

    #> initialise the list of relative NIfTI image CoMs
    com_nii = []

    #> modified affine for the centre of mass
    mA = imdct['affine'].copy()

    #> go through x, y and z
    for i in range(3):
        vox_size = max(imdct['affine'][i,:-1], key=abs)

        #> get the relative centre of mass for each axis (relative to the translation
        #> values in the affine matrix)
        if vox_size>0:
            com_rel = com[2-i] + imdct['affine'][i,-1]
        else:
            com_rel = com[2-i] - abs(vox_size)*imdct['shape'][-i-1] + imdct['affine'][i,-1]

        mA[i,-1] -= com_rel

        com_nii.append(com_rel)

    log.info('relative CoM values are:\n{}'.format(com_nii))
    #>------------------------------------------------------
    #> save to NIfTI
    innii = nib.load(imdct['fim'])

    #> get a new NIfTI image for the perturbed MR
    newnii = nib.Nifti1Image(innii.get_fdata(), mA, innii.header)

    fsplt = os.path.split(imdct['fim'])
    fnew = os.path.join(fsplt[0], fsplt[1].split('.nii')[0]+'_CoM-modified.nii.gz')

    #> save into a new file name for the T1w
    nib.save(newnii, fnew)
    #>------------------------------------------------------

    out = dict(
        fim=fnew,
        com_rel=com_nii,
        com_abs=com
        )

    return out
#===============================================================================




#> pick for now just one subject from XNAT database
sbj_lbl = 'BLSA-01850'

#> experiments available for the subject
exps = nixnat.get_list(
    xc['sbj']+'/' +sbj_lbl+ '/experiments?format=json',
    cookie=xc['cookie'])

#> experiment labels
elbls = [e['label'] for e in exps]

#> initialise lists of tuples of PET and MR visit number and experiment index
pets = []
mris = []

for i, e in enumerate(elbls):
    if 'PET' in e:
        v = np.int16(re.search('(?<=PET)\d*', e).group(0))
        pets.append((v, i))
    elif 'MR' in e:
        v = np.int16(re.search('(?<=MR)\d*', e).group(0))
        mris.append((v, i))

mri_vid = [s[0] for s in mris]

#> go through all PET visits and check if there is a corresponding MR visit.
for p in pets:
    pet_vst = p[0]
    if pet_vst in mri_vid:

        spth = os.path.join(opth, sbj_lbl+'_v'+ str(pet_vst))
        nimpa.create_dir(spth)

        #> get the PET PiB data
        petdct = nixnat.getscan(
            sbj_lbl,
            exps[p[1]]['ID'], #experiment label/ID
            xc,
            scan_ids = ['701', '800'], #PiB/O15
            dformat = 'NIFTI',
            outpath = spth,
            Cnt={'LOG':20},
            output_quality=False,
            # info_only=True,
            )

        #> and now the corresponding MRI T1w data
        ei = mris[mri_vid.index(pet_vst)][1]

        # #> get the MR data
        # t1dct = nixnat.getscan(
        #     sbj_lbl,
        #     exps[ei]['ID'], #experiment label/ID
        #     xc,
        #     scan_ids = ['100'], #PiB only
        #     dformat = 'NIFTI',
        #     outpath = spth,
        #     Cnt={'LOG':20},
        #     output_quality=False,
        #     # info_only=True,
        #     )
        # ft1 = t1dct.get([k for k in t1dct if '100' in k][0])[0]

        #-----------------------------------------------------------------------
        #> get the T1w-based brain ROI parcellations

        #> get the files needed
        rsrcs = nixnat.get_list(
            xc['sbj']+'/' +sbj_lbl+ '/experiments/' + exps[ei]['ID'] + '/resources/AMYPAD_GIF/files',
            cookie=xc['cookie']
        )

        #> create the output folder
        gif_path = os.path.join(spth, 'gif')
        nimpa.create_dir(gif_path)

        #> download the files
        gifdct = nixnat.getresources(
            rsrcs,
            xc,
            outpath = gif_path,
            cookie = xc['cookie'],
        )

        fprcl = [f for f in gifdct['nii'] if 'Parcellation' in f][0]
        ft1bc = [f for f in gifdct['nii'] if 'BiasCorrected' in f][0]
        fsgmt = [f for f in gifdct['nii'] if 'Segmentation' in f][0]
        #-----------------------------------------------------------------------

        for k in petdct:
            if k[:3].isnumeric():

                print(k)

                #petpth = petdct[k][0]

                ptrm = trim(petdct[k][0], fcomment='_'+k+'-PET_V'+str(pet_vst))
                #fsum = os.path.join(spth, k+'-PET_V'+str(pet_vst)+'_sum.nii.gz')

                #> get the file name of the average image for the dynamic PiB
                #> of the static image for the O15 image.
                if 'fsum' in ptrm:
                    imtdct = nimpa.getnii(ptrm['fsum'], output='all')
                    imcom = icom_correction(imtdct, Cnt={'LOG':20})
                    #> do the CoM adjustment for the dynamic
                    dyndct = nimpa.getnii(ptrm['fim'], output='all')
                    imcom_dyn = icom_correction(
                        dyndct,
                        Cnt={'LOG':20},
                        com=imcom['com_abs'])
                else:
                    imtdct = nimpa.getnii(ptrm['fim'], output='all')
                    imcom_dyn = {'fim': ''}
                    imcom = icom_correction(imtdct, Cnt={'LOG':20})





                spm_res = nimpa.coreg_spm(
                    imcom['fim'],
                    ft1bc,
                    fwhm_ref = fwhm_pet,
                    fwhm_flo = fwhm_mri,
                    #fwhm = [13,13],
                    costfun='nmi',
                    #fcomment = '',
                    outpath = os.path.dirname(imcom['fim']),
                    visual = 0,
                    save_txt = True,
                    save_arr = False,
                    del_uncmpr=True)

                #> resampled the parcellation, segmentation and T1w images and
                #> store in rpth
                rpth = os.path.join(os.path.dirname(imtdct['fim']), 'rsmpl')
                fout_t1 = os.path.join(rpth, k+'_V'+str(pet_vst)+'_resampled_T1w-BC.nii.gz')
                fout_pr = os.path.join(rpth, k+'_V'+str(pet_vst)+'_resampled_T1w-Par.nii.gz')

                ft1bc_r = nimpa.resample_spm(
                    imcom['fim'],
                    ft1bc,
                    spm_res['faff'],
                    intrp = 0.,
                    fimout=fout_t1,
                    del_ref_uncmpr = True,
                    del_flo_uncmpr = True,
                    del_out_uncmpr = True,)

                fprcl_r = nimpa.resample_spm(
                    imcom['fim'],
                    fprcl,
                    spm_res['faff'],
                    intrp = 0.,
                    fimout=fout_pr,
                    del_ref_uncmpr = True,
                    del_flo_uncmpr = True,
                    del_out_uncmpr = True,)

                print('''
                    \r==========================================================
                    \rFiles to be uploaded to XNAT:
                    \r1) Parcellation: {}
                    \r2) PET CoM: {}
                    \r3) PET CoM dynamic: {}
                    \r==========================================================
                    '''.format(fprcl_r, imcom['fim'], imcom_dyn['fim'])
                    )


                # sys.exit()

                #> upload data to XNAT
                res_url = xc['sbj']+'/' +sbj_lbl+ '/experiments/' +\
                          exps[p[1]]['ID'] + '/resources/PET_UPSAMPLED'

                #> prepare the container for trimmed and upsampled data.
                nixnat.put_data(
                    res_url+'?xsi:type=xnat:resourceCatalog&format=NIFTY',
                    cookie=xc['cookie'])

                #> upload dynamic upsampled and modified when present
                if not imcom_dyn['fim']=='':
                    nixnat.put_file(
                        res_url+'/files?content=PET',
                        imcom_dyn['fim'],
                        cookie=xc['cookie'])

                    nixnat.put_file(
                        res_url+'/files?content=PETAVG',
                        imcom['fim'],
                        cookie=xc['cookie'])
                else:
                    nixnat.put_file(
                        res_url+'/files?content=PET',
                        imcom['fim'],
                        cookie=xc['cookie'])

                #> upload parcellation image for the PET
                nixnat.put_file(
                    res_url+'/files?content=PRCL',
                    fprcl_r,
                    cookie=xc['cookie'])
