function [vringLm, ring2Lm, ring3Lm, ring4Lm] = computeRingsOnLandmarks(tri,lmIdx, ringN)


ring2Lm = [];
ring3Lm = [];
ring4Lm = [];

vring = compute_vertex_ring(tri);
disp('Computing Ring-1')
vringLm = vring(lmIdx);
% add original landmark
for i =1:length(vringLm)
    vringLm{i} = [vringLm{i} lmIdx(i)];
end

if ringN > 1
    % ring2
    disp('Computing Ring-2')

    ring2Lm = {};
    for i = 1:length(vringLm)
        ring = vringLm{i};
        r = [];
        for j = 1:length(ring)
            r = [r vring{ring(j)}];
        end
        ring2Lm{i} = unique(r);
    end
end

if ringN > 2
    %ring3
    disp('Computing Ring-3')

    ring3Lm = {};
    for i = 1:length(ring2Lm)
        ring = ring2Lm{i};
        r = [];
        for j = 1:length(ring)
            r = [r vring{ring(j)}];
        end
        ring3Lm{i} = unique(r);
    end
end

if ringN > 3
    %ring4
    disp('Computing Ring-4')

    ring4Lm = {};
    for i = 1:length(ring3Lm)
        ring = ring3Lm{i};
        r = [];
        for j = 1:length(ring)
            r = [r vring{ring(j)}];
        end
        ring4Lm{i} = unique(r);
    end
end
end

