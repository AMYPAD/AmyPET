"""Run centiloid pipeline level-2 for a new tracer FBB"""
__author__ = "Pawel J Markiewicz"
__copyright__ = "Copyright 2022"

import pickle
from pathlib import Path
from subprocess import run

import dcm2niix
from niftypet import nimpa

import amypet
from amypet import centiloid as centi

drv = Path('/data')
atlases = drv / 'AMYPET' / 'Atlas' / 'CL_2mm'
opth = drv / 'AMYPET' / 'CL' / 'FBB'

# ----------------------------------------------------------------------
# > convert to NIfTI as needed
if False:
    dirdata = drv / 'AMYPET' / 'CL' / 'FBB'
    grps = [x for x in dirdata.glob('*') if x.is_dir() if 'E-25' in str(x) or 'YC-10' in str(x)]

    for g in grps:

        print(f'======================\n{g.name}\n======================')

        nout = g / 'NIfTI'
        nimpa.create_dir(nout)
        dcmdir = [x for x in g.glob('*') if x.is_dir()]

        for s in dcmdir:

            run([dcm2niix.bin, '-i', 'y', '-v', 'n', '-o', nout, 'f', '%f_%s', s])
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# ELDRL
dirfbb = drv / 'AMYPET' / 'CL' / 'FBB' / 'E-25_FBB_90110' / 'NIfTI'
dirpib = drv / 'AMYPET' / 'CL' / 'FBB' / 'E-25_PiB_5070' / 'NIfTI'
dirmri = drv / 'AMYPET' / 'CL' / 'FBB' / 'E-25_MR' / 'NIfTI'
ffbbs = sorted(dirfbb.glob('*.nii'))
fpibs = sorted(dirpib.glob('*.nii'))
fmris = sorted(dirmri.glob('*.nii'))
# amypet.im_check_pairs(fpibs, fmris)
# amypet.im_check_pairs(ffbbs, fmris)

out_pe = centi.run(fpibs, fmris, atlases, tracer='pib', outpath=opth / 'output_pib_e')
with open(str(opth / 'output_pib_e.pkl'), 'wb') as f:
    pickle.dump(out_pe, f)

out_fe = centi.run(ffbbs, fmris, atlases, tracer='new', outpath=opth / 'output_fbb_e',
                   used_saved=True)
with open(str(opth / 'output_fbb_e.pkl'), 'wb') as f:
    pickle.dump(out_fe, f)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# YC
dirfbb = drv / 'AMYPET' / 'CL' / 'FBB' / 'YC-10_FBB_90110' / 'NIfTI'
dirpib = drv / 'AMYPET' / 'CL' / 'FBB' / 'YC-10_PiB_5070' / 'NIfTI'
dirmri = drv / 'AMYPET' / 'CL' / 'FBB' / 'YC-10_MR' / 'NIfTI'
ffbbs = sorted(dirfbb.glob('*.nii'))
fpibs = sorted(dirpib.glob('*.nii'))
fmris = sorted(dirmri.glob('*.nii'))
# amypet.im_check_pairs(fpibs, fmris)
# amypet.im_check_pairs(ffbbs, fmris)

out_py = centi.run(fpibs, fmris, atlases, tracer='pib', outpath=opth / 'output_pib_y')
with open(str(opth / 'output_pib_y.pkl'), 'wb') as f:
    pickle.dump(out_py, f)

out_fy = centi.run(ffbbs, fmris, atlases, tracer='new', outpath=opth / 'output_fbb_y')
with open(str(opth / 'output_fbb_y.pkl'), 'wb') as f:
    pickle.dump(out_fy, f)

# ----------------------------------------------------------------------

# ======================================================================
# ----------------------------------------------------------------------
with open(str(opth / 'output_pib_e.pkl'), 'rb') as f:
    out_pe = pickle.load(f)

with open(str(opth / 'output_fbb_e.pkl'), 'rb') as f:
    out_fe = pickle.load(f)

with open(str(opth / 'output_pib_y.pkl'), 'rb') as f:
    out_py = pickle.load(f)

with open(str(opth / 'output_fbb_y.pkl'), 'rb') as f:
    out_fy = pickle.load(f)
# ----------------------------------------------------------------------

# combine young and elderly dictionaries
out_pe.update(out_py)
out_fe.update(out_fy)
outp = out_pe
outf = out_fe

cal = amypet.calib_tracer(outp, outf)

# > save the transformations from UR to UR_PiB_Calc
Tur = amypet.save_ur2pib(cal, 'fbb')

# TEST
out_t = centi.run(fpibs[0], fmris[0], atlases, tracer='pib', outpath=opth / 'output_test_pib')
out_tt = centi.run(ffbbs[0], fmris[0], atlases, tracer='fbb', outpath=opth / 'output_test_fbb',
                   used_saved=True)

with open(opth / 'cal_data.pkl', 'wb') as f:
    pickle.dump(cal, f)

with open(opth / 'cal_data.pkl', 'rb') as f:
    cal_fbb = pickle.load(f)
