'''
Process CL and dynamic amyloid PET data
'''
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2022-3"


from pathlib import Path
from niftypet import nimpa
import spm12
import amypet
from amypet import backend_centiloid as centiloid
from amypet import params as Cnt


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INPUT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ignore_derived = False

# > Uptake ratio (UR) window def
ur_win_def=[5400,6600]

tracer = 'fbb' # 'pib', 'flute', 'fbp'


# > input PET folder
input_fldr = Path('/sdata/PNHS/FBB1')

outpath = input_fldr.parent/('d5_amypet_output_'+input_fldr.name) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CLASSIFY DICOM INPUT AND CONVERT TO NIfTI
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------------------------------
# > structural/T1w image
ft1w = amypet.get_t1(input_fldr, Cnt)

if ft1w is None:
    raise ValueError('Could not find the necessary T1w DICOM or NIfTI images')
#------------------------------

#------------------------------
# > processed the PET input data and classify it (e.g., automatically identify UR frames)
indat = amypet.explore_indicom(
    input_fldr,
    Cnt,
    tracer=tracer,
    find_ur=True,
    ur_win_def=ur_win_def,
    outpath=outpath/'DICOMs')

# > convert to NIfTIs
niidat = amypet.convert2nii(indat, use_stored=True)
#------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REMOVE ARTEFACTS AT THE END OF FOV
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
niidat = amypet.rem_artefacts(niidat, Cnt, artefact='endfov')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ALIGN PET FRAMES FOR STATIC/DYNAMIC IMAGING
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
aligned = amypet.align(
    niidat,
    Cnt,
    reg_tool='spm',
    ur_fwhm=4.5,
    use_stored=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CL QUANTIFICATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# > optional CSV file output for CL results 
# fcsv = input_fldr.parent/(input_fldr.name+'_AmyPET_CL.csv')

out_cl = centiloid.run(
    aligned['ur']['fur'],
    ft1w,
    Cnt,
    stage='f',
    voxsz=2,
    bias_corr=True,
    tracer=tracer,
    outpath=outpath/'CL',
    use_stored=True,
    #fcsv=fcsv
    )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dynamic PET and kinetic analysis

''' The CL intermediate results are used to move the atlas and 
    great matter mask to PET space. 
'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# > get all the regional data with atlas and GM probabilistic mask
dynvois = amypet.proc_vois(niidat, aligned, out_cl, outpath=niidat['outpath'].parent/'DYN')

# > kinetic analysis for a chosen volume of interest (VOI), here the frontal lobe
dkm = amypet.km_voi(dynvois,  voi='frontal', dynamic_break=True,  model='srtmb_basis', weights='dur')

# > voxel wise kinetic analysis
dimkm = amypet.km_img(
    aligned['fpet'],
    dynvois['voi']['cerebellum']['avg'],
    dynvois['dt'],
    dynamic_break=True,
    model='srtmb_basis',
    weights='dur',
    outpath=dynvois['outpath'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
# > show the great matter mask and atlas
nimpa.imscroll([dynvois['atlas_gm']['atlpet'], dynvois['atlas_gm']['gmpet']])

# > plot time activity curve for the cerebellum
plot(np.mean(dynvois['dt'],axis=0), dynvois['voi']['cerebellum']['avg'])

'''

# > images to show:
out_cl[next(iter(out_cl))]['fqc']

dkm['fig']