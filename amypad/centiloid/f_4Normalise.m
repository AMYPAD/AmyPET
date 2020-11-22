
function f_4Normalise(d_file,d_mri,d_pet)

spm_jobman('initcfg');
clear matlabbatch

matlabbatch{1}.spm.spatial.normalise.write.subj.def = {d_file};
matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {
                                                            d_mri
                                                            d_pet
                                                            };
matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [NaN NaN NaN
                                                          NaN NaN NaN];
matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';


spm_jobman('run',matlabbatch);
end

