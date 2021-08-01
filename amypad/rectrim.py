import os

from niftypet import nimpa, nipet

# import scipy.ndimage as ndi


# definition of dynamic frames for kinetic analysis
frmdef = ["def", [4, 15], [8, 30], [9, 60], [2, 180], [8, 300]]

# get all the constants and LUTs
mMRpars = nipet.get_mmrparams()

# ------------------------------------------------------
folderin = folderin = "/data/amyloid_brain"
# folderin = '/store/downloads/1946/S00151_18715520/TP0'

# recognise the input data as much as possible
datain = nipet.classify_input(folderin, mMRpars)
# ------------------------------------------------------

# switch on verbose mode
mMRpars["Cnt"]["VERBOSE"] = True

# output path
opth = os.path.join(datain["corepath"], "output_dyn")


# -----------------------
hst = nipet.mmrhist(datain, mMRpars)

# offset for the time from which meaningful events are detected
toff = nipet.lm.get_time_offset(hst)

# dynamic frame timings
frm_timings = nipet.lm.dynamic_timings(frmdef, offset=toff)

nipet.lm.draw_frames(hst, frm_timings["timings"])
# -----------------------

# -----------------------------------------------------------------------
# >MU-MAPS
# -----------------------------------------------------------------------
# > hardware mu-map
muhdct = nipet.hdw_mumap(datain, [1, 2, 4], mMRpars, outpath=opth, use_stored=True)


# > object mu-map with alignment
mupdct = nipet.align_mumap(
    datain,
    mMRpars,
    outpath=opth,
    store=True,
    hst=hst,
    itr=2,
    petopt="ac",
    fcomment="_mu",
    musrc="pct",
)

# > object mu-map without alignment--straight from DICOM resampled to PET
# muodct = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)
# -----------------------------------------------------------------------

recon = nipet.mmrchain(
    datain,
    mMRpars,
    frames=frm_timings["timings"],
    mu_h=muhdct,
    mu_o=mupdct,  # muodct,
    itr=5,
    fwhm=0.0,
    outpath=opth,
    fcomment="_dyn",
    store_img=True,
    store_img_intrmd=True,
)

# -----------------------------------------------------------------------
# > END of generating (recon) data
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# > Trimming and upscaling
# -----------------------------------------------------------------------

# > get the file data
dyndir = os.path.join(opth, "PET", "multiple-frames")
dyndir_fs = os.listdir(dyndir)

fdyns = [os.path.join(dyndir, f) for f in dyndir_fs if "itr5" in f and "dyn_i.nii" in f]

# sort the dynamic images and output the sorted file names and numpy array
imdyn = nimpa.niisort(fdyns)

# > perform trimming and upscaling; note that the first row is 'blank' due to the offset
dtrm = nimpa.imtrimup(imdyn["files"][1:], store_img_intrmd=True)

# > trimmed and scaled images:
dtrm["fimi"]
