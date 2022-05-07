''' Run centiloid pipeline level-2 for a new tracer FLUTE.
'''

__author__ = "Pawel J Markiewicz"
__copyright__ = "Copyright 2022"

import os, sys
from pathlib import Path
import pickle
import openpyxl

from niftypet import nimpa
from amypet import centiloid as centi
import amypet
import spm12

import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress


drv = Path('/sdata/AMYPET')
atlases = drv/Path('Atlas/CL_2mm/')
opth = drv/Path('CL/FLUTE')

#----------------------------------------------------------------------
# AD
dirflt = drv/Path('CL/FLUTE/GE_AD_F18_NIFTI')
dirpib = drv/Path('CL/FLUTE/GE_AD_C11_NIFTI')
dirmri = drv/Path('CL/FLUTE/GE_AD_MRI_NIFTI')
fflts = [dirflt/f for f in dirflt.glob('*.nii')]
fpibs = [dirpib/f for f in dirpib.glob('*.nii')]
fmris = [dirmri/f for f in dirmri.glob('*.nii')]
fflts.sort()
fpibs.sort()
fmris.sort()

'''
amypet.im_check_pairs(fpibs, fmris)
amypet.im_check_pairs(fflts, fmris)
'''


out_pe = centi.run(fpibs, fmris, atlases, tracer='pib', outpath=opth/'output_pib_a') #, used_saved=True
with open(str(opth/'output_pib_a.pkl'), 'wb') as f:
    pickle.dump(out_pe, f)


out_fe = centi.run(fflts, fmris, atlases, tracer='new', outpath=opth/'output_flt_a')
with open(str(opth/'output_flt_a.pkl'), 'wb') as f:
    pickle.dump(out_fe, f)
#----------------------------------------------------------------------


#----------------------------------------------------------------------
# YC
dirflt = drv/Path('CL/FLUTE/GE_YHV_F18_NIFTI')
dirpib = drv/Path('CL/FLUTE/GE_YHV_C11_NIFTI')
dirmri = drv/Path('CL/FLUTE/GE_YHV_MRI_NIFTI')
fflts = [dirflt/f for f in dirflt.glob('*.nii')]
fpibs = [dirpib/f for f in dirpib.glob('*.nii')]
fmris = [dirmri/f for f in dirmri.glob('*.nii')]
fflts.sort()
fpibs.sort()
fmris.sort()

'''
amypet.im_check_pairs(fpibs, fmris)
amypet.im_check_pairs(fflts, fmris)
'''

out_py = centi.run(fpibs, fmris, atlases,  tracer='pib', outpath=opth/'output_pib_y')
with open(str(opth/'output_pib_y.pkl'), 'wb') as f:
    pickle.dump(out_py, f)

out_fy = centi.run(fflts, fmris, atlases, tracer='new', outpath=opth/'output_flt_y')
with open(str(opth/'output_flt_y.pkl'), 'wb') as f:
    pickle.dump(out_fy, f)
#----------------------------------------------------------------------



#======================================================================
#----------------------------------------------------------------------
with open(str(opth/'output_pib_a.pkl'), 'rb') as f:
    out_pa = pickle.load(f)

with open(str(opth/'output_flt_a.pkl'), 'rb') as f:
    out_fa = pickle.load(f)

with open(str(opth/'output_pib_y.pkl'), 'rb') as f:
    out_py = pickle.load(f)

with open(str(opth/'output_flt_y.pkl'), 'rb') as f:
    out_fy = pickle.load(f)
#----------------------------------------------------------------------

# combine young and elderly dictionaries
out_pa.update(out_py)
out_fa.update(out_fy)
outp = out_pa
outf = out_fa

cal = amypet.calib_tracer(outp, outf)

# > save the transformations from SUVr to SUVr_PiB_Calc
Tsuvr = amypet.save_suvr2pib(cal, 'flute')


# TEST
out_t  = centi.run(fpibs[0], fmris[0], atlases, tracer='pib', outpath=opth/'output_test_pib')
out_tt = centi.run(fflts[0], fmris[0], atlases, tracer='flute', outpath=opth/'output_test_flt')