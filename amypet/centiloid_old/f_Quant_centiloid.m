function [fnbase, greyCerebellum, wholeCerebellum, wholeCerebellumBrainStem, pons] = f_Quant_centiloid(s_PET_dir, dir_RR)

d_roi_cort = [dir_RR, filesep, 'voi_ctx_2mm.nii'];
d_roi_cergy = [dir_RR, filesep, 'voi_CerebGry_2mm.nii'];
d_roi_pons = [dir_RR, filesep, 'voi_Pons_2mm.nii'];
d_roi_whcer = [dir_RR, filesep, 'voi_WhlCbl_2mm.nii'];
d_roi_cerbst = [dir_RR, filesep, 'voi_WhlCblBrnStm_2mm.nii'];

v_roi_cort = spm_vol(d_roi_cort);
v_roi_cergy = spm_vol(d_roi_cergy);
v_roi_pons = spm_vol(d_roi_pons);
v_roi_whcer = spm_vol(d_roi_whcer);
v_roi_cerbst = spm_vol(d_roi_cerbst);

m_roi_cort = spm_read_vols(v_roi_cort);
m_roi_cergy = spm_read_vols(v_roi_cergy);
m_roi_pons = spm_read_vols(v_roi_pons);
m_roi_whcer = spm_read_vols(v_roi_whcer);
m_roi_cerbst = spm_read_vols(v_roi_cerbst);

ind_cergy = (m_roi_cergy == 1);
ind_pons = (m_roi_pons == 1);
ind_whcer = (m_roi_whcer == 1);
ind_cerbst = (m_roi_cerbst == 1);
ind_cort = (m_roi_cort == 1);

fnbase = cell(1, length(s_PET_dir));
greyCerebellum = cell(1, length(s_PET_dir));
wholeCerebellum = cell(1, length(s_PET_dir));
wholeCerebellumBrainStem = cell(1, length(s_PET_dir));
pons = cell(1, length(s_PET_dir));

fprintf(2, 'Subject %d/%d done\r', 0, length(s_PET_dir));
for i_subj = 1:length(s_PET_dir)
    d_PET = s_PET_dir{i_subj};
    [~, name, ~] = fileparts(d_PET);

    f_noNaN(d_PET);
    v_pet = spm_vol(d_PET);
    m_pet = spm_read_vols(v_pet);
    v_cort = mean(m_pet(ind_cort));

    fnbase{1, i_subj} = name;
    greyCerebellum{1, i_subj} = v_cort / mean(m_pet(ind_cergy));
    wholeCerebellum{1, i_subj} = v_cort / mean(m_pet(ind_whcer));
    wholeCerebellumBrainStem{1, i_subj} = v_cort / mean(m_pet(ind_cerbst));
    pons{1, i_subj} = v_cort / mean(m_pet(ind_pons));

    fprintf(2, 'Subject %d/%d done\r', i_subj, length(s_PET_dir));
end
fprintf(2, '\n');
