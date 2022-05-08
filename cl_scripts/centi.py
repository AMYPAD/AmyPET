''' Run calibration centiloid pipeline for PiB
'''

__author__ = "Pawel J Markiewicz"
__copyright__ = "Copyright 2022"

import os, sys
from pathlib import Path
import pickle

from niftypet import nimpa
from amypet import centiloid as centi
import amypet

# > input paths
drv = Path('/data/AMYPET')
atlases = drv/Path('Atlas/CL_2mm/')
opth = drv/Path('CL/PiB')

#----------------------------------------------------------------------
# AD
dirpet = drv/Path('CL/PiB/AD-100_PET_5070/nifti')
dirmri = drv/Path('CL/PiB/AD-100_MR/nifti')
fpets = [os.path.join(dirpet, f) for f in os.listdir(dirpet)]
fmris = [os.path.join(dirmri, f) for f in os.listdir(dirmri)]
fpets.sort()
fmris.sort()

# > run visual check of the images before running the CL pipeline
#amypet.im_check_pairs(fpets, fmris)

flip_pet = len(fpets)*[(1,1,1)]
flip_pet[9]  = (1,-1,1)
flip_pet[22] = (1,-1,1)
flip_pet[26] = (1,-1,1)
flip_pet[34] = (1,-1,1)
flip_pet[36] = (1,-1,1)
flip_pet[40] = (1,-1,1)
flip_pet[44] = (1,-1,1)

out_ad = centi.run(fpets, fmris, atlases, flip_pet=flip_pet, outpath=opth/'output_pib_ad')
with open(str(opth/'output_pib_ad.pkl'), 'wb') as f:
    pickle.dump(out_ad, f)
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# YC
dirpet = drv/Path('CL/PiB/YC-0_PET_5070/nifti')
dirmri = drv/Path('CL/PiB/YC-0_MR/nifti')
fpets = [os.path.join(dirpet, f) for f in os.listdir(dirpet)]
fmris = [os.path.join(dirmri, f) for f in os.listdir(dirmri)]
fpets.sort()
fmris.sort()

# > run visual check of the images before running the CL pipeline
#amypet.im_check_pairs(fpets, fmris)

out_yc = centi.run(fpets, fmris, atlases, outpath=opth/'output_pib_yc')
with open(str(opth/'output_pib_yc.pkl'), 'wb') as f:
    pickle.dump(out_yc, f)
#----------------------------------------------------------------------



#----------------------------------------------------------------------
with open(str(opth/'output_pib_yc.pkl'), 'rb') as f:
    out_yc = pickle.load(f)

with open(str(opth/'output_pib_ad.pkl'), 'rb') as f:
    out_ad = pickle.load(f)
#----------------------------------------------------------------------

refs = amypet.get_clref(opth/'SupplementaryTable1.xlsx')

diff = amypet.check_suvrs(out_yc, out_ad, refs)

diff = amypet.check_cls(out_yc, out_ad, diff, refs)

#cla = amypet.save_cl_anchors(diff, outpath=Path('/home/pawel/Dropbox'))
cla = amypet.save_cl_anchors(diff)