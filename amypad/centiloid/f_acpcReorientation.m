
function f_acpcReorientation(imglist)
% batch script for AC-PC reorientation
% This script tries to set AC-PC with 2 steps.
% 1. Set origin to center (utilizing a script by F. Yamashita)
% 2. Coregistration of the image to icbm152.nii under spm/toolbox/DARTEL
% 
% K. Nemoto 22/May/2017

%% Initialize batch
spm_jobman('initcfg');
matlabbatch = {};

%% Select images
%imglist=spm_select(Inf,'image','Choose MRI you want to set AC-PC');

%% Set the origin to the center of the image
% This part is written by Fumio Yamashita.
for i=1:size(imglist,1)
    file = deblank(imglist(i,:));
    st.vol = spm_vol(file);
    vs = st.vol.mat\eye(4);
    vs(1:3,4) = (st.vol.dim+1)/2;
    spm_get_space(st.vol.fname,inv(vs));
end

%% Prepare the SPM window
% interactive window (bottom-left) to show the progress, 
% and graphics window (right) to show the result of coregistration 

%spm('CreateMenuWin','on'); %Comment out if you want the top-left window.
spm('CreateIntWin','on');
spm_figure('Create','Graphics','Graphics','on');

%% Coregister images with icbm152.nii under spm12/toolbox/DARTEL
for i=1:size(imglist,1)
    matlabbatch{i}.spm.spatial.coreg.estimate.ref = {fullfile(spm('dir'),'toolbox','DARTEL','icbm152.nii,1')};
    matlabbatch{i}.spm.spatial.coreg.estimate.source = {deblank(imglist(i,:))};
    matlabbatch{i}.spm.spatial.coreg.estimate.other = {''};
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{i}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
end

%% Run batch
%spm_jobman('interactive',matlabbatch);
spm_jobman('run',matlabbatch);

