
function v_quant=f_Quant_centiloid(s_PET_dir,dir_RR,dir_quant)


d_roi_cort=[dir_RR filesep 'voi_ctx_2mm.nii'];
d_roi_cergy=[dir_RR filesep 'voi_CerebGry_2mm.nii'];
d_roi_pons=[dir_RR filesep 'voi_Pons_2mm.nii'];
d_roi_whcer=[dir_RR filesep 'voi_WhlCbl_2mm.nii'];
d_roi_cerbst=[dir_RR filesep 'voi_WhlCblBrnStm_2mm.nii'];

v_roi_cort=spm_vol(d_roi_cort);
v_roi_cergy=spm_vol(d_roi_cergy);
v_roi_pons=spm_vol(d_roi_pons);
v_roi_whcer=spm_vol(d_roi_whcer);
v_roi_cerbst=spm_vol(d_roi_cerbst);

m_roi_cort=spm_read_vols(v_roi_cort);
m_roi_cergy=spm_read_vols(v_roi_cergy);
m_roi_pons=spm_read_vols(v_roi_pons);
m_roi_whcer=spm_read_vols(v_roi_whcer);
m_roi_cerbst=spm_read_vols(v_roi_cerbst);

v_ref=zeros(4,1);
v_quant=cell(length(s_PET_dir)+1,5);

ind_cergy=(m_roi_cergy==1);
ind_pons=(m_roi_pons==1);
ind_whcer=(m_roi_whcer==1);
ind_cerbst=(m_roi_cerbst==1);
ind_cort=(m_roi_cort==1);

v_quant{1,1}='RR';
v_quant{1,2}='GreyCerebellum';
v_quant{1,5}='Pons';
v_quant{1,3}='WholeCerebellum';
v_quant{1,4}='WholeCerebellumBrainStem';


for i_subj=1:length(s_PET_dir)

    d_PET=[s_PET_dir(i_subj).folder filesep 'w' s_PET_dir(i_subj).name]; 
   
    f_noNaN(d_PET);
    v_pet=spm_vol(d_PET);
    m_pet=spm_read_vols(v_pet);

    v_ref(1)=mean(m_pet(ind_cergy));
    v_ref(4)=mean(m_pet(ind_pons));
    v_ref(2)=mean(m_pet(ind_whcer));
    v_ref(3)=mean(m_pet(ind_cerbst));

    v_cort=mean(m_pet(ind_cort));
    v_quant{i_subj+1,1}=s_PET_dir(i_subj).name;

    for i=1:4
    
        v_quant{i_subj+1,i+1}=v_cort/v_ref(i);
    
    end
    
    fprintf(1,['Subject ' num2str(i_subj) ' done\n']);

end

save([dir_quant filesep 'Quant_' date '.mat'],'v_quant')


