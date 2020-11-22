%This function changes NaN's of the image to zeros

%Inputs:
    %i1=Image
    %i2=New name (optional)
    
%Outputs:
    %o1=Initial image without NaN's. Name: initial name or new name
    %(optional)

function f_noNaN(image,varargin)

V1=spm_vol(image);
M1=spm_read_vols(V1);

M1(isnan(M1)==1)=0;

if nargin>1
    
    V1.fname=varargin{1};
    
end

spm_write_vol(V1,M1);