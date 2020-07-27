function [minErrRing, errRingAll] = estimateRingError(gtLandmarks,modFinal,lmRing)

minErrRing = zeros(length(lmRing),1);
errRingAll = {};
for i = 1:length(lmRing)
    ring = lmRing{i};
    err_ring = [];
    for j = 1:length(ring)
        e = diag(pdist2(gtLandmarks(i,:), modFinal(ring(j),:),...
            'euclidean'));
        err_ring = [err_ring; e];
    end
    minErrRing(i) = min(err_ring);
    errRingAll{i,1} = err_ring;
end