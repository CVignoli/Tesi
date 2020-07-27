function [A S R t] = estimatePose(landModel,landImage)
%% Pose Estimation
baricM = mean(landModel,1);
P = landModel - repmat(baricM,size(landModel,1),1);

baricI = mean(landImage,1);
p = landImage - repmat(baricI,size(landImage,1),1);

P = P';
p = p';
qbar = baricI';
Qbar = baricM';

A = p*pinv(P);

[S R] = qr(A');
rr=S;
S=R';
R=rr';
t = qbar - A*Qbar;
end