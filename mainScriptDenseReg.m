% --------------------
addpath(genpath('./'))
% Optimization params
derr = 0.01;
maxIter = 30;
lambda_all  = 1;

% Debugs
debugMesh = true;

gtLandmarksPath = 'data/landmarks/';
landmarkList = dir([gtLandmarksPath '*groundtruth.ldmk']);
landmarkList = rmfield(landmarkList, {'date', 'folder', 'bytes', 'isdir', 'datenum'});
landmarkList = struct2cell(landmarkList); 

meshPath = 'data/models/';
meshList = dir([meshPath '*.mat']);
meshList = rmfield(meshList, {'date', 'folder', 'bytes', 'isdir', 'datenum'});
meshList = struct2cell(meshList);
    
load('data/SLC_50_1_1.mat')
Components_res = reshape_components(Components);     

% Load 3DMM and Landmarks -------------------------------------------------
load('data/avgModel_bh_1779_NE.mat')
% Zero mean 3D model and landmarks ----------------------------------------
idxLandmarks3D(1:17) = [];
landmarks3D(1:17,:) = [];
baric_avg = mean(avgModel,1);
avgModel = avgModel - repmat(baric_avg,size(avgModel,1),1);
landmarks3D = landmarks3D - repmat(baric_avg,size(landmarks3D,1),1);
     
% Load GT landmarks configuration -----------------------------------------
load(['data/landmarksFRGC_CVPR20_ver2.mat']);

lm3dmm = idxLandmarks3D;
lm3dmm_all = lm3dmm;
lm3dmmGT = frgcLm_buLips;
lm3dmmGT_all = lm3dmmGT;

% Compute ring-1 on landmarks ---------------------------------------------
vring = compute_vertex_ring(compute_delaunay(avgModel));
vring = vring(lm3dmmGT_all);
% -------------------------------------------------------------------------

errLm_final = [];
err_final = [];
missingModels = {};
% ------------------------------------------------------------------------

for i = 1:length(meshList)
    disp(['Processing model ' num2str(i)])
    try
        load([meshPath meshList{i}]);
        % Find ground-truth landmarks -------------------------------------
        idx = find(contains(landmarkList, meshList{i}(1:end-4)));
        lmTGT = readGTlandmarks([gtLandmarksPath landmarkList{idx}]);
        lmTGT(end,:) = [];
        % Select Landmarks and models Configuration
        lm3dmm = idxLandmarks3D;
        lm3dmm_all = lm3dmm;
        lm3dmm = lm3dmm([20 23 26 29 32 38 35 41 14]);
        
        % Check if target mesh has less points than 3DMM
        if size(vertex,1) < size(avgModel,1)
            disp([num2str(i) ' model with less vertices!'])
            continue;            
        end
        % -----------------------------------------------------------------
        
        % Zero mean GT model
        baric = mean(vertex,1);
        modGT = vertex - repmat(baric,size(vertex,1),1);
        % -----------------------------------------------------------------                
                
        % Find closest vertex in gt model for annotation error
        lmTGT = lmTGT - repmat(baric,size(lmTGT,1),1);
        d = pdist2(modGT, lmTGT, 'euclidean');
        [mindists, lmidxGT] = min(d);
        lmTGT = modGT(lmidxGT,:);
        lmidxGT_all = lmidxGT;
        % -----------------------------------------------------------------
         
        % Initialization ----------------------------------------
        defShape = avgModel;
        
        % Initial ICP -----------------------------------------------------
        [Ricp, Ticp] = icp(defShape', modGT', 15,...
            'Matching', 'kDtree','Minimize','plane', 'Extrapolation', true,'Verbose',false);
        modGT = (Ricp * (modGT') + repmat(Ticp, 1, size(modGT,1)))';
        % -----------------------------------------------------------------
        
        % Find noseTip
        nt = find(modGT(:,3) == max(modGT(:,3)));
        ntTrasl = avgModel(lm3dmmGT(6),:) - modGT(nt,:);
        modGT = modGT + ntTrasl;
        % -----------------------------------------------------------------
        
        % Refine  ICP -----------------------------------------------------
        [Ricp, Ticp] = icp(defShape', modGT', 15,...
            'Matching', 'kDtree','Minimize','plane', 'Extrapolation', true,'Verbose',false);
        modGT = (Ricp * (modGT') + repmat(Ticp, 1, size(modGT,1)))';
        % -----------------------------------------------------------------
                                                          
        
        % Initial Association ---------------------------------------------  
        [modPerm,err,minidx,missed] = bidirectionalAssociation(modGT, defShape);
        err_init = err;
        errLm_init = estimateRingError(modGT(lmidxGT_all,:),modPerm,vring);
        
        % Re-align
        iidx = setdiff(1:size(defShape,1),missed);
        [A, S, R, trasl] = estimatePose(modGT(minidx(iidx),:),defShape(iidx,:));
        modPerm = getProjectedVertex(modPerm,S,R,trasl)';
        modGT = getProjectedVertex(modGT,S,R,trasl)';
        disp(['Mean distance initialization: ' num2str(err)])
        % -----------------------------------------------------------------
        
        % NRF -------------------------------------------------------------
        disp('Start NRF routine')
        d = 1;
        t = 1;                
        alphas = []; % Keep for Recognition        
        while t < maxIter && d > derr          
            % Fit the 3dmm
            alpha  = alphaEstimation_fast_3D(defShape,modPerm,...
                Components_res,1:6704,Weights,lambda_all);
            defShape = deform_3D_shape_fast(defShape',...
                Components,alpha)';
            
            % Re-associate points as average
            [modPerm,errIter,minidx,missed] = bidirectionalAssociation(modGT, defShape);
            
            d = abs(err - errIter);
            err = errIter;
            
            err_lm = mean(diag(pdist2(modGT(lmidxGT_all,:),...
                modPerm(lm3dmmGT_all,:),...
                'euclidean')));
            
            disp(['Mean distance: ' num2str(err) ' - Mean Landmark error - ' num2str(err_lm)])
            % Iterate
            t=t+1;
            alphas = [alphas alpha];
        end
        % -----------------------------------------------------------------

        % Debug -----------------------------------------------------------
        if debugMesh
            figure
            subplot(1,2,1)
            plot_landMesh(defShape,defShape(lm3dmmGT,:),0)
            title('NRF')
            subplot(1,2,2)
            plot_landMesh(modGT,modGT(lmidxGT,:),0)
            title('GT model')
            pause;
            close
        end
        disp('Done.')
        % -----------------------------------------------------------------
        
        % Registered GT model building ------------------------------------
        disp('Start Dense Registration routine')                                                                              
        modFinal = reassociateDuplicates(modGT, defShape);
        % ------------
        disp('Done!')
        
        err_lm = mean(diag(pdist2(modGT(lmidxGT_all,:),...
            modFinal(lm3dmmGT_all,:),...
            'euclidean')));
        disp(['Mean Landmark error - ' num2str(err_lm)])
        
        % Debug -----------------------------------------------------------
        if debugMesh
            figure
            subplot(1,2,1)
            plot_landMesh(modFinal,modFinal(lm3dmmGT,:),0)            
            title('Registered Model')
            subplot(1,2,2)
            plot_landMesh(modGT,modGT(lmidxGT_all,:),0)
            title('GT model')
            figure
            subplot(1,2,1)
            plot_landModel(modFinal,modFinal(lm3dmmGT,:),'r.',1)
            view([0 90])
            title('Registered Model')
            subplot(1,2,2)
            plot_landModel(modGT,modGT(lmidxGT_all,:),'r.',1)
            view([0 90])
            title('GT model')
            pause;
            close
        end
        % -----------------------------------------------------------------
        
        % Compute Final Error ---------------------------------------------
        errLm_final = estimateRingError(modGT(lmidxGT_all,:),modFinal,vring);
        % -----------------------------------------------------------------

    catch ME
        errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
            ME.stack(1).name, ME.stack(1).line, ME.message);
        fprintf(1, '%s\n', errorMessage);
        missingModels = [missingModels; meshList{i}];
    end
    disp(['Model ' num2str(i) ' out of ' num2str(length(meshList))])
    
end





