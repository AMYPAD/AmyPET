"""Resources file for AmyPET Constants and Parameters"""
__author__ = "Pawel J. Markiewicz"
__copyright__ = "Copyright 2023"


Cnt = dict(
    # > registration parameters in the centiloid pipeline
    regpars=dict(

        # > smoothing parameter (FWHM) for the T1-to-MNI space registration (applies only to T1w image)
        fwhm_t1_mni=3,

        # > smoothing parameters (FWHM) for the PET-to-T1 (or T1-to-PET) space registration for PET and T1w
        fwhm_t1=3,
        fwhm_pet=6,

        # > cost function for the rigid body registration
        costfun='nmi',

        # > if True, will use the SPM visuals
        visual=False),

    # > artefact correction
    endfov_corr=dict(

        # > frame range for which the correction is applied
        frm_rng=(0,10),

        # > the axial (z) voxel margin where performing correction at the ends of FOV
        z_margin=10),

    # > alignment of PET frames
    align=dict(

        # > minimal duration of PET frame to be considered for registration
        frame_min_dur=60,

        # > decay correction for the coffee-break protocol if used and requested
        decay_corr=False,

        # > registration cost function for the alignment
        reg_costfun='nmi',

        # > image smoothing prior to registration for alignment
        reg_fwhm=8,

        # > metric threshold for applying registration
        reg_thrshld=2.0,),
    )