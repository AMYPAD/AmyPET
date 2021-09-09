
import SimpleITK as sitk
import sys, os
import os, sys
import shutil
import glob
from subprocess import run
from niftypet import nimpa

import multiprocessing
nthrds = multiprocessing.cpu_count()


#====================================
def rungif(fimin, gifpath=None, outpath=None):
    '''
    fimin   - NIfTI image input file (T1w MR image)
    gifpath - path to the GIF folders with executables and database

    '''

    if gifpath is None or not os.path.exists(gifpath):
        raise ValueError('wrong path to GIF.')

    if not os.path.isfile(fimin) or not 'nii' in fimin:
        raise ValueError('incorrect input image file.')


    gifexe = os.path.join(gifpath, 'bin', 'seg_GIF')
    gifdb = os.path.join(gifpath, 'db', 'db.xml')


    #---------------------------------------------
    # > create outputs
    if outpath is None:
        opth = os.path.join(os.path.dirname(fimin), 'out')
    else:
        opth = outpath
    nimpa.create_dir(opth)
    logerr = os.path.join(opth, 'err.log')
    logout = os.path.join(opth, 'out.log')
    #---------------------------------------------

    #---------------------------------------------
    gifresults = run(
        [gifexe, 
            '-in', fimin,
            '-db', gifdb,
            '-v', '1', '-regNMI', '-segPT', '0.1',
            '-out', opth,
            '-temper', '0.05', '-lncc_ker', '-4', '-omp', str(nthrds), '-regBE', '0.001', '-regJL', '0.00005'
        ],
        capture_output=True)
    #---------------------------------------------


    #---------------------------------------------
    # > output logs
    with open(logerr, 'wb') as f:
        f.write(gifresults.stderr)
    with open(logout, 'wb') as f:
        f.write(gifresults.stdout)
    #---------------------------------------------

    return gifresults
#====================================



'''
import os
import glob
from subprocess import run

fimin = glob.glob(os.path.join(gpth, 'N4bias', '*N4bias*.nii*'))[0]
fimin = '/home/pawel/cs_nifty/DPUK_dl/py_test/TP0/DICOM_MPRAGE_20200226150442_15_N4bias.nii.gz'
fimin = '/home/pawel/cs_nifty/DPUK_dl/py_test/NEW002_PETMR_V1_00015_MR_images_MPRAGE_MPRAGE_20200212145346_15.nii'
'''


gpth = '/data/DPUK/TRT'
for d in os.listdir(gpth):
    spth = os.path.join(gpth,d)
    if os.path.isdir(spth):
        for tp in os.listdir(spth):
            tpth = os.path.join(spth,tp)

            # > check if MPRAGE DICOM folder is present
            fglb = [f for f in glob.glob(os.path.join(tpth, '*MPRAGE*')) if os.path.isdir(f)]
            for fldr in fglb:
                # > run dcm2niix to convert the DICOMs into NIfTI
                fnii = [f for f in glob.glob(os.path.join(tpth, '*MPRAGE*')) if os.path.isfile(f) and f.endswith(('.nii', 'nii.gz'))]
                if len(fnii)==0:
                    run(
                        ['/home/pawel/NiftyPET_tools/dcm2niix/bin/dcm2niix',
                         '-i', 'y',
                         '-v', 'n',
                         '-o', tpth,
                         'f', '%f_%s',
                         fldr])

            fn4b = glob.glob(os.path.join(tpth, 'N4bias', '*N4bias*.nii*'))
            fgif = glob.glob(os.path.join(tpth, 'GIF', '*Parcellation*.nii*'))
            
            if len(fn4b)<1:
                fnii = [f for f in glob.glob(os.path.join(tpth, '*MPRAGE*')) if os.path.isfile(f) and f.endswith(('.nii', 'nii.gz'))]
                print(f'N4bias input: {fnii}')
                biascorr = nimpa.bias_field_correction(
                    fnii[0],
                    executable='sitk',
                    outpath=tpth)
                fingif = biascorr['fim']

            elif len(fn4b)==1 and len(fgif)==0:
                fingif = fn4b[0]
                #print(f'i> found N4-bias corrected {fingif}')

            else: continue


            # > make use that SimpleITK is installed: conda install -c simpleitk simpleitk
            print(f'i> running GIF on {fingif}')
            rungif(fingif, gifpath='/home/pawel/AMYPET/GIF2BBRC', outpath=os.path.join(tpth, 'GIF'))


#nohup python amypet_gif.py > ~/amypet_gif.log 2>&1 &