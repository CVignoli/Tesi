% Created by Iacopo Masi <masi>@dsi.unifi.it
% Copyright MICC - Media Integration and Communication Center
% and USC Computer Vision Lab.
%
% P = rotatePointCloud(P,R,t)
%
% The function performs rigid rotation and traslation to a pointclouds and
% outputs the result as P. The rotation is performed on the baricenter of
% the pointcloud.
%
% Note that if t is [] empty the function performs only rotation.
%
% Input
%
% R: rotation matrix 3x3
% t: translation vector 3x1
%
% Output
%
% coordTex: P rotated pointcloud

function P = rotatePointCloud(P,R,t,centerToOrigin)
if centerToOrigin
   baric = mean(P,1);
   P = P - repmat(baric,size(P,1),1);
end
P = R*P';
if ~isempty(t)
   P = P + repmat(t',size(P,2),1)';
end
if 0
   P = P + repmat(baric,size(P,2),1)';
end
return