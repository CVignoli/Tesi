import _3DMM
import Matrix_operations as mo
import util_for_graphic as ufg
#import matlab.engine
#eng = matlab.engine.start_matlab()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import icp
import glob
import mat73
import scipy.io
import h5py
import numpy as np
from scipy.spatial.distance import cdist
import numpy.matlib as npm
import sys
import os


def bidirectionalAssociation(modGT, defShape):
    D = cdist(modGT, defShape)
    [mindists, minidx] = np.min(D)

    [mindistsGT, minidxGT] = np.min(D, [], 2)
    threshGlobal = np.mean(mindistsGT) + np.std(mindistsGT)
    toRemGlobal = mindistsGT > threshGlobal
    [unGT, _, _] = np.unique(minidxGT)

    modPerm = np.zeros(np.size(defShape))
    for i in range(len(unGT)):
        kk = np.nonzero(mindistsGT == unGT[i])
        thresh = np.mean(mindistsGT[kk]) + np.std(mindistsGT[kk])
        toRem = kk[mindistsGT[kk] > thresh]
        toRem = [toRem, toRemGlobal]
        kk = np.setdiff1d(kk, toRem)
        if len(kk) > 1:
            modPerm[unGT[i], :] = np.mean(modGT[kk, :])
        elif len(kk) == 1:
            modPerm[unGT[i], :] = modGT[kk, :]

    missed = np.nonzero(np.sum(modPerm, 2) == 0)
    modPerm[missed, :] = modGT[minidx[missed], :]

    err = np.mean(mindists)
    return modPerm, err, minidx, missed


def reassociateDuplicates(modGT, defShape):
    D = cdist(modGT, defShape)
    # Case for average model
    if np.size(modGT) == np.size(defShape):
        D = D + np.identity(np.size(D)) * sys.float_info.max
    [mindists, minidx] = np.min(D)
    [un, iidx, iun] = np.unique(minidx)

    iidxMiss = np.setdiff1d(np.size(defShape, axis=0), iidx)  # indici vanno bene??? Vedremo :)
    df = defShape[iidx, :]

    modTmp = modGT
    modPerm = []
    modPerm[iidx, :] = modTmp[un, :]

    while df:
        # Remove unique vertices from gt
        modTmp[un, :] = []
        # Re-compute distances
        D = cdist(modTmp, df)
        [_, minidx] = np.min(D)
        [un, iidx, _] = np.unique(minidx)

        # Store new unique indices and remove from the old unique set
        iidx_n = iidxMiss[iidx]
        iidxMiss[iidx] = []
        df[iidx, :] = modTmp[un, :]

    err = np.mean(mindists)
    return modPerm, err


mat_op = mo.Matrix_op
_3DM = _3DMM._3DMM

sys.path.insert(1, './')

# Optimization params
derr = 0.01
maxIter = 30
lambda_all = 1

# Debugs
debugMesh = True

gtLandmarksPath = 'data/landmarks/'
landmarkList = os.listdir(gtLandmarksPath)
landmarkList = [f for f in landmarkList if f.endswith('.groundtruth.ldmk')]

meshPath = 'data/models/'
meshList = os.listdir(meshPath)
meshList = [f for f in meshList if f.endswith('.mat')]
meshList2 = []
for i in range(len(meshList)):
    meshList2 = meshList2.__add__([meshList[i]])
    meshList2[i] = meshList2[i][:-4]

slc = mat73.loadmat('data/SLC_50_1_1.mat')
components = slc.get('Components')
aligned_models_data = None
components_R = mat_op(components, aligned_models_data)
Components_res = components_R.X_res

# Load 3DMM and Landmarks
avgM = mat73.loadmat('data/avgModel_bh_1779_NE.mat')  # guardare cosa c'è dentro ...
avgModel = avgM.get('avgModel')

model1 = scipy.io.loadmat('data/models/02463d546.mat')
model2 = scipy.io.loadmat('data/models/04395d192.mat')
model3 = scipy.io.loadmat('data/models/04681d145.mat')
models = [model1, model2, model3]

# Zero mean 3D model and landmarks
idxLandmarks3D = avgM.get('idxLandmarks3D')
idxLandmarks3D = np.delete(idxLandmarks3D, slice(17), 0)

landmarks3D = avgM.get('landmarks3D')
landmarks3D = np.delete(landmarks3D, slice(17), 0)
baric_avg = np.mean(avgModel, axis=0)
avgModel = avgModel - npm.repmat(baric_avg, np.size(avgModel, axis=0), 1)
landmarks3D = landmarks3D - npm.repmat(baric_avg, np.size(landmarks3D, axis=0), 1)

# Load GT landmarks configuration
frgcLm_buLips = mat73.loadmat('data/landmarksFRGC_CVPR20_ver2.mat')

lm3dmm = idxLandmarks3D
lm3dmm_all = lm3dmm
lm3dmmGT = frgcLm_buLips
lm3dmmGT_all = lm3dmmGT

# Compute ring-1 on landmarks
# compute_vertex_ring = scipy.io.loadmat('toolboxes/toolbox_graph/compute_vertex_ring.m')   #come chiamare metodo matlab? engine, link desktop (da fare dopo)
# compute_delaunay = scipy.io.loadmat('toolboxes/toolbox_graph/compute_delaunay.m')
# vring = compute_vertex_ring(compute_delaunay(avgModel))   # scipy.spatial.delaunay ... qualcosa di simile

errLm_final = []
err_final = []
missingModels = {}

for i in range(len(meshList)):
    print('Processing model ' + str(i))
    try:
        mlist = meshList[i]  # load meshlist
        # Find ground-truth landmarks
        idx = np.nonzero(meshList2[i] in landmarkList)    # non lo so come restituire indice dove meshlist2 si trova in landmarkList

        # lmTGT = readGTlandmarks            # reimplementare (?) ??
        lmTGT = np.zeros((14, 3))             #cazzata ma dal readGT deve uscire 14x3
        lmTGT = np.delete(lmTGT, 13, 0)
        # Select Landmarks and models Configuration
        lm3dmm = idxLandmarks3D
        lm3dmm_all = lm3dmm
        b = [19, 22, 25, 28, 31, 37, 34, 40, 13]
        lm3dmm = [lm3dmm[i] for i in b]

        # Check if target mesh has less points than 3DMM
        vertex = models[0].get('vertex')                               # metti i alla fine (tengo per confronto MLab)

        if np.size(vertex, axis=0) < np.size(avgModel, axis=0):
            print(str(i) + ' model with less vertices!')
            continue
        # ....................................

        # Zero mean GT model

        baric = np.mean(vertex, axis=0)
        #modGT = vertex - npm.repmat(baric, np.size(lmTGT, axis=0), 1) finchè non risolvo readGT si va poco lontano...
        # ..........................................
        """
        # Find closest vertex in gt model for annotation error
        lmTGT = lmTGT - npm.repmat(baric, np.size(lmTGT, axis=0), 1)
        d = cdist((modGT, lmTGT))
        [mindists, lmidxGT] = np.min(d)
        lmTGT = modGT[lmidxGT, :]
        lmidxGT_all = lmidxGT

        # Initialization
        defShape = avgModel

        # Initialize ICP
        [Ricp, Ticp] = []  # ICP ???????
        modGT = (Ricp * (modGT) + npm.repmat(Ticp, 1, np.size(modGT, axis=0)))  # (rivedere trasposte) adattare in base alla trasformazione

        # Initial Association
        [modPerm, err, minidx, missed] = bidirectionalAssociation(modGT, defShape)

        err_init = []
        errLm_init = estimateringerror  # va implementata?

        # Re-align

        iidx = np.setdiff1d(np.size(defShape, axis=0), missed)
        [A, S, R, trasl] = _3DMM._3DMM.estimate_pose(modGT[minidc[iidx], :], defShape[iidx, :])
        modPerm = _3DMM._3DMM.getProjectedVertex(modPerm, S, R, trasl)  # ' che significa?? TRASPOSTO
        modGT = _3DMM._3DMM.getProjectedVertex(modGT, S, R, trasl)  # Controlla trasposto sopra e sotto
        print('Mean distance initialization: ' + str(err))
        # ..........................................................

        # NRF ....................................
        print('Start NRF routine')
        d = 1
        t = 1
        alphas = []  # Keep for Recognition
        while t < maxIter and d > derr:
            # Fit the 3dmm
            alpha = _3DMM._3DMM.alphaEstimation(defShape, modPerm)  # dove li prendo altri parametri?

            defShape = _3DM.deform_3D_shape_fast(np.transpose(defShape), components, alpha)  # Da trasporre

            # Re-associate points as average
            [modPerm, errIter, minidx, missed] = bidirectionalAssociation(modGT, defShape)

            d = np.abs(err - errIter)
            err = errIter

            err_lm = np.mean(np.diag(cdist(modGT[lmidxGT_all, :], modPerm[lm3dmmGT_all, :])))

            print('Mean distance: ' + str(err) + ' - Mean Landmark error - ' + str(err_lm))

            # Iterate
            t = t + 1
            # alphas = [alphas alpha]   # che significa??
        # ...................

        # Debug .............
        if debugMesh:
            plt.figure()
            plt.subplot(1, 2, 1)
            # plot_landMesh dove è implementata?
            plt.title('NRF')
            plt.subplot(1, 2, 2)
            # plot_landMesh
            plt.title('GT model')
            plt.pause()

        print('Done.')
        # ........................

        # Registered GT model building ...............
        print('Start Dense Registration routine')
        modFinal = reassociateDuplicates(modGT, defShape)
        # ............
        print('Done!')

        err_lm = np.mean(np.diag(cdist(modGT[lmidxGT_all, :], modFinal[lm3dmmGT_all, :])))
        print('Mean Landmark error - ' + str(err_lm))

        # Debug ......................
        if debugMesh:
            plt.figure()
            plt.subplot(1, 2, 1)
            # plot_landMesh ??
            plt.title('Registered Model')
            plt.subplot(1, 2, 2)
            # plot_landMesh ??
            plt.title('GT model')
            plt.figure()
            plt.subplot(1, 2, 1)
            # plot_landModel   #dove si trova sta funzione?
            # view() non ho davvero idea di come chiamarlo...
            plt.title('Registered Model')
            plt.subplot(1, 2, 2)
            # plot_landmodel
            # view() ??????
            plt.title('GT model')
            plt.pause()
            plt.close()
        # ..................

        # Compute Final Error .............
        errLm_final = estimateRingError  # va implementata (riga 181)
        # ....................
        """
    except:
        errorMessage = print('Error somewhere')

        """
        missingModels = [missingModels, meshList[i]]

    print('Model ' + str(i) + 'out of ' + str(len(meshList)))
        """
