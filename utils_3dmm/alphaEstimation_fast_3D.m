function [Alpha]  = alphaEstimation_fast_3D(landModel,tgtLand,Components_res,idxLandmarks,Weights,lambda)

% Compute the alpha parameters based on the reprojection error of the
% landmarks
%landModel = bsxfun(@minus,landModel, mean(landModel));
%tgtLand = bsxfun(@minus,tgtLand, mean(tgtLand));
X = tgtLand - landModel;
X = X(:);
Y = zeros(size(X,1),size(Components_res,3));
for c=1:size(Components_res,3)
    comp = Components_res(idxLandmarks,:,c);
    Y(:,c)=comp(:);
end
if lambda == 0
    Alpha = Y\X;
else
    invW = diag(lambda./(diag(Weights)));
    if size(invW,2) ~= 1
        invW = diag(invW);
    end
    YY = ( Y'*Y + diag(invW) )^-1;
    Alpha = YY*Y'*X;
end
return