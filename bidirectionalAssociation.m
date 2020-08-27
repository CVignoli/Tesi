function [modPerm, err, minidx, missed ]= bidirectionalAssociation(modGT, defShape)

D = pdist2(modGT, defShape, 'euclidean');
[mindists, minidx] = min(D);

[mindistsGT, minidxGT] = min(D,[],2);
threshGlobal = mean(mindistsGT) + std(mindistsGT);
toRemGlobal = mindistsGT > threshGlobal;
[unGT, ~, ~] = unique(minidxGT);

modPerm = zeros(size(defShape));
for i = 1:length(unGT)
    kk = find(minidxGT==unGT(i));
    thresh = mean(mindistsGT(kk)) + std(mindistsGT(kk));
    toRem = kk(mindistsGT(kk) > thresh);
    toRem = [toRem; toRemGlobal];
    kk = setdiff(kk,toRem);
    if length(kk) > 1
        modPerm(unGT(i),:) = mean(modGT(kk,:));
    elseif length(kk) == 1
        modPerm(unGT(i),:) = modGT(kk,:);
    end
    
end

missed = find(sum(modPerm,2)==0);
modPerm(missed,:) = modGT(minidx(missed),:);

err = mean(mindists);