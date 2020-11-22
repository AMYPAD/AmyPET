

dir_spm='/home/gsalvado/Software/spm12';%MODIFY: SPM directory
dir_code='/home/gsalvado/Code/ALFA_PET/BBRC_Centiloid_Pipeline';%MODIFY: Code directory
dir_MRI='/home/gsalvado/Projects/ALFA_PET';%MODIFY: MRI directory
dir_PET='/home/gsalvado/Projects/ALFA_PET';%MODIFY: PET directory
dir_RR='/home/gsalvado/Atlas/CL_2mm';%MODIFY: Reference regions ROIs directory (They must be the standard Centiloid RR that you can download in the GAAIN Centioid website, in 2mm and in nifti format)
dir_quant='/home/gsalvado/Projects/ALFA_PET/Quant_realigned';%MODIFY:Quantification directory

s_PET_dir=dir([dir_PET filesep '*_PET.nii']);%MODIFY: PET images list
s_MRI_dir=dir([dir_MRI filesep '*_MRI.nii']);%MODIFY: MR images lsit

n_subj_PET=length(s_PET_dir);% Number of PET images
n_subj_MRI=length(s_MRI_dir);% Number of MR images

if n_subj_PET ~=n_subj_MRI
    
    error('ERROR: Different number of PET and MR images');
    
end

%parpool(6);%MODIFY: Number of workers in parallel

tic
for i_subj=1:n_subj_PET
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       STEP 0: Reorient images           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    d_PET=[s_PET_dir(i_subj).folder filesep s_PET_dir(i_subj).name];
    f_acpcReorientation(d_PET)
    
    fprintf(1,['Reorient PET subject ' num2str(i_subj) ' done\n'])
    
    d_MRI=[s_MRI_dir(i_subj).folder filesep s_MRI_dir(i_subj).name];
    f_acpcReorientation(d_MRI)
    
    fprintf(1,['Reorient MRI subject ' num2str(i_subj) ' done\n'])
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       STEP 1: CorregisterEstimate       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    f_1CorregisterEstimate(d_MRI,dir_spm)
    fprintf(1,['Step 1: Patient ' num2str(i_subj) ' done\n' ])
    
    %% Check Reg

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       STEP 2: CorregisterEstimate      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    f_2CorregisterEstimate(d_MRI,d_PET)
    fprintf(1,['Step 2: Patient ' num2str(i_subj) ' done\n' ])
    
%% Check Reg

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       STEP 3: Segment       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    f_3Segment(d_MRI,dir_spm)
    fprintf(1,['Step 3: Patient ' num2str(i_subj) ' done\n' ])
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       STEP 4: Normalise       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    d_file_norm=[dir_MRI filesep 'y_' s_MRI_dir(i_subj).name];
    f_4Normalise(d_file_norm,d_MRI,d_PET)
    fprintf(1,['Step 4: Patient ' num2str(i_subj) ' done\n' ])
    
end

toc

s_PET=dir([dir_PET filesep 'w*PET.nii']); %MODIFY
f_Quant_centiloid(s_PET,dir_RR,dir_quant);