function f_2CorregisterEstimate(d_mri, d_pet)

spm_jobman('initcfg');
clear matlabbatch

matlabbatch{1}.spm.spatial.coreg.estimate.ref = {d_mri};
matlabbatch{1}.spm.spatial.coreg.estimate.source = {d_pet};
matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4, 2];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7, 7];

spm_jobman('run', matlabbatch);
end
