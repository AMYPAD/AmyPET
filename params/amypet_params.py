"""Resources file for AmyPET Constants and Parameters"""
__author__ = "Pawel J. Markiewicz"
__copyright__ = "Copyright 2023"




Cnt = dict(
    pttrn_t1 = ['mprage', 't1', 't1w', 'spgr'],

    tracer_names = {
        'pib': ['pib'],
        'flute': ['flt', 'flut', 'flute', 'flutemetamol'],
        'fbb': ['fbb', 'florbetaben'],
        'fbp': ['fbp', 'florbetapir']
    },

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

    timings=dict(
        # > SUVr time window post injection and duration for amyloid tracers
        suvr_twindow = {
            'pib': [90 * 60, 110 * 60, 1200],
            'flute': [90 * 60, 110 * 60, 1200],
            'fbb': [90 * 60, 110 * 60, 1200],
            'fbp': [50 * 60, 60 * 60, 600]},
                
        # > break time for coffee break protocol (target)
        break_time = 1800,

        # > time margin for the 1st coffee break acquisition
        breakdyn_t = (1200, 2400),

        # > minimum time for the full dynamic acquisition
        fulldyn_time = 3600,

        # > margin used for accepting SUVr time windows (0.1 corresponds to 10%)
        margin = 0.1,
        ),
        
    )