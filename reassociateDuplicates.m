function [modPerm, err ]= reassociateDuplicates(modGT, defShape)

D = pdist2(modGT, defShape, 'euclidean');
% Case for average model
if size(modGT) == size(defShape)
    D = D + eye(size(D))*realmax;
end
[mindists, minidx] = min(D);
[un, iidx, iun] = unique(minidx);

iidxMiss = setdiff(1:size(defShape,1),iidx);
df = defShape(iidxMiss,:);

modTmp = modGT;
modPerm = [];
modPerm(iidx,:) = modTmp(un,:);

while ~isempty(df)
    
    % Remove unique vertices from gt
    modTmp(un,:) = [];
    
    % Re-compute distances
    D = pdist2(modTmp, df, 'euclidean');
    [~, minidx] = min(D);
    [un, iidx, ~] = unique(minidx);
    
    % Store new unique indices and remove from the old unique set    
    iidx_n = iidxMiss(iidx);
    iidxMiss(iidx) = [];
    df(iidx, :) = [];
    
    % Add to the new model
    modPerm(iidx_n,:) = modTmp(un,:);
    
end

err = mean(mindists);