function [Alpha]  = alphaEstimation_fast(landImage,projLandModel,Components_res,idxLandmarks,S,R,t,Weights,lambda)

% Compute the alpha parameters based on the reprojection error of the
% landmarks
% REMOVE MEAN??
%landImage = bsxfun(@minus,landImage, mean(landImage));
%projLandModel = bsxfun(@minus,projLandModel, mean(projLandModel));

X = landImage - projLandModel;
X = X(:);
Y = zeros(size(X,1),size(Components_res,3));
for c=1:size(Components_res,3)
    vertexOnImComp = getProjectedVertex(Components_res(idxLandmarks,:,c),S,R,t)';
    Y(:,c)=vertexOnImComp(:);
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