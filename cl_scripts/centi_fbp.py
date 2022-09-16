"""Run centiloid pipeline level-2 for a new tracer FBB"""
__author__ = "Pawel J Markiewicz"
__copyright__ = "Copyright 2022"

import os
import pickle
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from subprocess import run

import dcm2niix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import spm12
from niftypet import nimpa
from scipy.stats import linregress

import amypet
from amypet import centiloid as centi

drv = Path('/data')
atlases = drv / 'AMYPET' / 'Atlas' / 'CL_2mm'
dirdata = drv / 'AMYPET' / 'CL' / 'FBP'
opth = dirdata

#----------------------------------------------------------------------
# > convert to NIfTI as needed
if False:
    intrm_fldr = 'NIfTI_converted_intrm'
    final_fldr = 'NIfTI'

    # > folder to be corrected (inconsistency detected upsetting the running of AmyPET)
    dircorr = dirdata / 'Young_Control_PiB' / '411884_PiB3_AC'
    if (dircorr / '4118844_PiB3_AC').is_dir():
        os.rename(dircorr / '4118844_PiB3_AC', dircorr / '411884_PiB3_AC')

    grps = [x for x in dirdata.glob('*') if x.is_dir() if 'Elder' in str(x) or 'Young' in str(x)]

    for g in grps:
        print(f'======================\n{g.name}\n======================')

        dcmall = g.glob('**/*.dcm')
        dcmdirs = np.unique([d.parent for d in dcmall])
        # dreps = [d.parent.parent.parent for d in dcmdirs]
        # # > DICOM folder counter per subject
        # dcnt = Counter(dreps)§

        # > all DICOMs per subject dictionary
        sDCM = {}
        for d in dcmdirs:
            prnt = d.parent.parent.parent
            if not prnt in sDCM:
                sDCM[prnt] = list(d.glob('*.dcm'))
            else:
                sDCM[prnt] += list(d.glob('*.dcm'))

        for s in sDCM:

            #------------------------------------
            print(f'i> copying files for subject: {s}')
            tmpdir = opth / 'tmp' / s.name
            nimpa.create_dir(tmpdir)

            for f in sDCM[s]:
                shutil.copy(f, tmpdir)

            print('i> convert DICOMs to NIfTI file...')
            npth = opth / intrm_fldr / g.name / s.name
            nimpa.create_dir(npth)

            run([
                dcm2niix.bin,
                              #'-i', 'y',
                '-v',
                'n',
                '-o',
                npth,
                'f',
                '%f_%s',
                tmpdir])

            # > remove the temporary DICOM files
            shutil.rmtree(tmpdir)
            #------------------------------------

            #------------------------------------
            # > find all NIfTI files per subject and combine them when needed
            spth = opth / final_fldr / g.name
            nimpa.create_dir(spth)

            tmpim = None
            for fn in npth.glob('*.nii'):
                ndct = nimpa.getnii(fn, output='all')
                if tmpim is None:
                    tmpim = ndct['im']
                else:
                    tmpim += ndct['im']

            # > in case of 4D NIfTI image, sum it here
            if len(tmpim.shape) > 3:
                tmpim = np.sum(tmpim, axis=0)

            fnm = re.findall(r'\d+[\w\t]*', s.name)[0]
            fout = spth / (fnm+'.nii.gz')

            nimpa.array2nii(
                tmpim.astype(np.float32), ndct['affine'], fout,
                trnsp=(ndct['transpose'].index(0), ndct['transpose'].index(1),
                       ndct['transpose'].index(2)), flip=ndct['flip'])
            #------------------------------------

# #==============\
# fnii = Path('/data/AMYPET/CL/FBP/NIfTI_converted/Young_Control_PiB/101437_PiB3_AC/101437_PiB3_AC_unnamed_20120101142541_574880.nii')
# ndct = nimpa.getnii(fnii, output='all')
# tmpim = ndct['im']
# tmpim = np.sum(tmpim, axis=0)
# fout = fnii.parent/('cc.nii.gz')
# #==============\

#----------------------------------------------------------------------

#----------------------------------------------------------------------
# ELDRL
dirfbp = drv / 'AMYPET' / 'CL' / 'FBP' / 'NIfTI' / 'Elder_subject_florbetapir'
dirpib = drv / 'AMYPET' / 'CL' / 'FBP' / 'NIfTI' / 'Elder_subject_PiB'
dirmri = drv / 'AMYPET' / 'CL' / 'FBP' / 'NIfTI' / 'Elder_subject_MRI'
ffbps = sorted(dirfbp.glob('*.nii*'))
fpibs = sorted(dirpib.glob('*.nii*'))
fmris = sorted(dirmri.glob('*.nii*'))
# amypet.im_check_pairs(fpibs, fmris)
# amypet.im_check_pairs(ffbps, fmris)

out_pe = centi.run(fpibs, fmris, atlases, tracer='pib', outpath=opth / 'output_pib_e')
with open(str(opth / 'output_pib_e.pkl'), 'wb') as f:
    pickle.dump(out_pe, f)

out_fe = centi.run(ffbps, fmris, atlases, tracer='new', outpath=opth / 'output_fbp_e',
                   used_saved=True)
with open(str(opth / 'output_fbp_e.pkl'), 'wb') as f:
    pickle.dump(out_fe, f)
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# YC
dirfbp = drv / 'AMYPET' / 'CL' / 'FBP' / 'NIfTI' / 'Young_Control_florbetapir'
dirpib = drv / 'AMYPET' / 'CL' / 'FBP' / 'NIfTI' / 'Young_Control_PiB'
dirmri = drv / 'AMYPET' / 'CL' / 'FBP' / 'NIfTI' / 'Young_Control_MRI'
ffbps = sorted(dirfbp.glob('*.nii*'))
fpibs = sorted(dirpib.glob('*.nii*'))
fmris = sorted(dirmri.glob('*.nii*'))
# amypet.im_check_pairs(fpibs, fmris)
# amypet.im_check_pairs(ffbps, fmris)

out_py = centi.run(fpibs, fmris, atlases, tracer='pib', outpath=opth / 'output_pib_y',
                   used_saved=True)
with open(str(opth / 'output_pib_y.pkl'), 'wb') as f:
    pickle.dump(out_py, f)

out_fy = centi.run(ffbps, fmris, atlases, tracer='new', outpath=opth / 'output_fbp_y')
with open(str(opth / 'output_fbp_y.pkl'), 'wb') as f:
    pickle.dump(out_fy, f)

#----------------------------------------------------------------------

sys.exit()

#======================================================================
#----------------------------------------------------------------------
with open(str(opth / 'output_pib_e.pkl'), 'rb') as f:
    out_pe = pickle.load(f)

with open(str(opth / 'output_fbp_e.pkl'), 'rb') as f:
    out_fe = pickle.load(f)

with open(str(opth / 'output_pib_y.pkl'), 'rb') as f:
    out_py = pickle.load(f)

with open(str(opth / 'output_fbp_y.pkl'), 'rb') as f:
    out_fy = pickle.load(f)
#----------------------------------------------------------------------

# combine young and elderly dictionaries
out_pe.update(out_py)
out_fe.update(out_fy)
outp = out_pe
outf = out_fe

cal = amypet.calib_tracer(outp, outf)

# > save the transformations from SUVr to SUVr_PiB_Calc
Tsuvr = amypet.save_suvr2pib(cal, 'fbp')


# TEST
out_t = centi.run(fpibs[0], fmris[0], atlases, tracer='pib', outpath=opth / 'output_test_pib')
out_tt = centi.run(ffbbs[0], fmris[0], atlases, tracer='fbb', outpath=opth / 'output_test_fbb',
                   used_saved=True)

with open(opth / 'cal_data.pkl', 'wb') as f:
    pickle.dump(cal, f)

with open(opth / 'cal_data.pkl', 'rb') as f:
    cal_fbp = pickle.load(f)


'''
TESTS FOR CHECKING CONSISTENCY WITH PROVIDED DATA
=======
>>>>>>> 2c5df0134c48bf9fad58528eeac341da8ecc8d6a
import openpyxl as xl

info = xl.load_workbook(dirdata / 'Avid_Centiloid_standard_method.xlsx')
dat = info['Sheet1']
istart = 2

pid = [i.value for i in dat['A'][istart:]]
suvr_wc_fbp = np.array([i.value for i in dat['D'][istart:]])
suvr_wc_pib = [i.value for i in dat['E'][istart:]]
suvr_wc_cnv = [i.value for i in dat['G'][istart:]]
cl_wc_fbp = np.array([i.value for i in dat['I'][istart:]])
cl_wc_pib = [i.value for i in dat['J'][istart:]]

# > index conversion from the xlsx file to the AmyPET indexes (sorted)
idxs = [pid.index(int(i)) for i in cal['wc']['sbj']]
suvrf_avid = suvr_wc_fbp[idxs]
suvrf_amyp = cal['wc']['calib']['cl_suvr'][:, 2]

clf_avid = cl_wc_fbp[idxs]

figure()
plot(suvrf_avid, suvrf_amyp, '.')

<<<<<<< HEAD
fig, ax  = plt.subplots()
ax.scatter(cal[rvoi]['calib']['cl_suvr'][:,1], cal[rvoi]['calib']['cl_suvr'][:,2], c='black')
amypet.aux.identity_line(ax=ax, ls='--', c='b')
'''
=======
fig, ax = plt.subplots()
ax.scatter(cal[rvoi]['calib']['cl_suvr'][:, 1], cal[rvoi]['calib']['cl_suvr'][:, 2], c='black')
amypet.aux.identity_line(ax=ax, ls='--', c='b')
>>>>>>> 2c5df0134c48bf9fad58528eeac341da8ecc8d6a
