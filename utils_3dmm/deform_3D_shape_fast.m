function [ def_shape ] = deform_3D_shape_fast( mean_face, eigenvecs, alpha)

%DEFORM_3D_SHAPE Construct a new 3D face shape as S = m + sum_1:n(alpha_i*w_i)
%deforming the mean face shape
%   mean_face = vertices matrix of the mean face shape
%   eigenvecs = principal components
%   alpha = coefficients of the morphing

dim = size(eigenvecs,1)/3;
%% sum_1:n(alpha_i*w_i): alpha is column vector

alpha_full = repmat(alpha',size(eigenvecs,1),1);
tmp_eigen = alpha_full.*eigenvecs;
sumVec = sum(tmp_eigen,2);
sumMat = reshape(sumVec',3,dim);

%%  S = m + sum_1:n(alpha_i*w_i)
def_shape = mean_face + sumMat;

end

