from pathlib import Path
from typing import List

import matplotlib
import notebook as notebook

import _3DMM
import Matrix_operations as mo
import util_for_graphic as ufg
import matlab.engine

from matlab_utils import convert_ndarray_to_matlab

eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath(r"toolboxes"))
eng.addpath(eng.genpath(r"utils_gen"))
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import copy
import mat73
import scipy.io
import open3d as o3d
import h5py
import numpy as np
from scipy.spatial.distance import cdist
import numpy.matlib as npm
import sys
import os

#prova portatile git

def remove_extensions(input_list: List) -> List:
    output_list = []

    for i in range(len(input_list)):
        output_list.append(input_list[i])
        output_list[i] = output_list[i].stem.split(".")[0]

    return output_list

def bidirectionalAssociation(modGT, defShape):
    D = cdist(modGT, defShape)
    mindists = np.amin(D, axis=0)
    minidx = np.argmin(D, axis=0)
    mindistsGT = np.amin(D, axis=1)
    minidxGT = np.argmin(D, axis=1)
    threshGlobal = np.mean(mindistsGT) + np.std(mindistsGT)
    toRemGlobal = mindistsGT > threshGlobal
    idxGlobal = np.where(toRemGlobal)[0]
    unGT = np.unique(minidxGT)

    modPerm = np.zeros(np.shape(defShape))
    for i in range(len(unGT)):
        kk = np.where(minidxGT == unGT[i])
        kk = np.asarray(kk)
        thresh = np.mean(mindistsGT[kk]) + np.std(mindistsGT[kk])
        toRem = kk[mindistsGT[kk] > thresh]
        toRem = np.concatenate((toRem, idxGlobal))
        kk = np.setdiff1d(kk, toRem)
        if len(kk) > 1:
            modPerm[unGT[i], :] = np.mean(modGT[kk, :], axis=0)
        elif len(kk) == 1:
            modPerm[unGT[i], :] = modGT[kk, :]

    missed = np.nonzero(np.sum(modPerm, 1) == 0)
    missed = np.asarray(missed)
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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    #o3d.visualization.draw_geometries([source_temp, target_temp], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])
    o3d.visualization.draw_geometries([source_temp, target_temp])


mat_op = mo.Matrix_op
_3DMM = _3DMM._3DMM()

# sys.path.insert(1, './')

# Optimization params
derr = 0.01
maxIter = 30
lambda_all = 1

# Debugs
debugMesh = True

gtLandmarksPath = 'data/landmarks/'
landmarkList = list(Path(gtLandmarksPath).glob("*.groundtruth.ldmk"))

meshPath = 'data/models/'
meshList = list(Path(meshPath).glob("*.mat"))

slc = mat73.loadmat('data/SLC_50_1_1.mat')
components = slc.get('Components')
weights = slc.get('Weights')
aligned_models_data = None
components_R = mat_op(components, aligned_models_data)
Components_res = components_R.X_res

# Load 3DMM and Landmarks
avgM = mat73.loadmat('data/avgModel_bh_1779_NE.mat')
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
frgcLm_buLips_gen = mat73.loadmat('data/landmarksFRGC_CVPR20_ver2.mat')
frgcLm_buLips = frgcLm_buLips_gen.get('frgcLm_buLips')

lm3dmm = idxLandmarks3D-1
lm3dmm_all = lm3dmm
lm3dmmGT = frgcLm_buLips-1
lm3dmmGT_all = lm3dmmGT
lm3dmmGT_all_vring = lm3dmmGT_all

# Compute ring-1 on landmarks
avgModelMatlab = matlab.double(avgModel.tolist())
vring = eng.compute_vertex_ring(eng.compute_delaunay(avgModelMatlab))
vring = np.array(vring)
vring = vring[lm3dmmGT_all_vring.astype(int)]

errLm_final = []
err_final = []
missingModels = {}
basenameMeshList = remove_extensions(meshList)
basenameLandmarkList = remove_extensions(list(Path(gtLandmarksPath).glob("*.groundtruth.ldmk")))

for i in range(len(meshList)):
    print('Processing model ' + str(i))
    mlist = meshList[i]
    # Find ground-truth landmarks
    basenameMeshList = remove_extensions(meshList)

    idx = basenameLandmarkList.index(basenameMeshList[i])

    lmTGT, _ = eng.readGTlandmarks(str(landmarkList[idx]), nargout=2)
    lmTGT = np.array(lmTGT._data.tolist()).reshape((3, 14)).T
    lmTGT = np.delete(lmTGT, 13, 0)
    # Select Landmarks and models Configuration
    lm3dmm = idxLandmarks3D -1
    lm3dmm_all = lm3dmm
    b = [19, 22, 25, 28, 31, 37, 34, 40, 13]
    lm3dmm = [lm3dmm[i] for i in b]

    # Check if target mesh has less points than 3DMM
    vertex = models[i].get('vertex')

    if np.size(vertex, axis=0) < np.size(avgModel, axis=0):
        print(f'{i}  model with less vertices!')
        continue
    # ....................................

    # Zero mean GT model

    baric = np.mean(vertex, axis=0)
    modGT = vertex - npm.repmat(baric, np.size(vertex, axis=0), 1)
    # ..........................................

    # Find closest vertex in gt model for annotation error
    lmTGT = lmTGT - npm.repmat(baric, np.size(lmTGT, axis=0), 1)
    d = cdist(modGT, lmTGT)
    lmidxGT = np.argmin(d, axis=0)
    lmTGT = modGT[lmidxGT, :]
    lmidxGT_all = lmidxGT

    # Initialization

    defShape = avgModel

    # Initial ICP
    dShape = o3d.geometry.PointCloud()
    dShape.points = o3d.utility.Vector3dVector(defShape)
    o3d.io.write_point_cloud("dati pc/defShape.ply", dShape)

    mGT = o3d.geometry.PointCloud()
    mGT.points = o3d.utility.Vector3dVector(modGT)
    o3d.io.write_point_cloud("dati pc/modGT.ply", mGT)
    icp_result = o3d.registration.registration_icp(mGT, dShape, 15)
    #draw_registration_result(mGT, dShape, icp_result.transformation)
    transf_vec = np.asarray(icp_result.transformation)
    Ricp = transf_vec[0:3, 0:3]
    Ticp = transf_vec[0:3, 3]
    modGT = np.transpose(np.matmul(Ricp, modGT.T) + np.transpose(npm.repmat(Ticp, np.size(modGT, axis=0), 1)))
    """
    #see for results
    
    mGT1 = o3d.geometry.PointCloud()
    mGT1.points = o3d.utility.Vector3dVector(modGT)
    o3d.io.write_point_cloud("dati pc/modGT1.ply", mGT1)
    #draw_registration_result(mGT1, dShape, icp_result.transformation)
    """

    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = [defShape[:, 0]]
    y = [defShape[:, 1]]
    z = [defShape[:, 2]]

    ax.scatter3D(x,y,z, color='r')


    x1 = [modGT[:, 0]]
    y1 = [modGT[:, 1]]
    z1 = [modGT[:, 2]]

    ax.scatter3D(x1, y1, z1, color='b')
    plt.show()
    """
    # Find noseTip
    
    nt = np.where(modGT[:, 2] == np.max(modGT[:, 2]))
    ntTrasl = avgModel[int(lm3dmmGT[5]), :] - modGT[int(nt[0]), :]
    modGT = modGT + ntTrasl
    
    #Refine ICP
    mGT1 = o3d.geometry.PointCloud()
    mGT1.points = o3d.utility.Vector3dVector(modGT)
    o3d.io.write_point_cloud("dati pc/modGT1.ply", mGT1)
    icp_result = o3d.registration.registration_icp(mGT1, dShape, 15)
    # draw_registration_result(mGT, dShape, icp_result.transformation)
    transf_vec = np.asarray(icp_result.transformation)
    Ricp = transf_vec[0:3, 0:3]
    Ticp = transf_vec[0:3, 3]
    modGT = np.transpose(np.matmul(Ricp, modGT.T) + np.transpose(npm.repmat(Ticp, np.size(modGT, axis=0), 1)))
    mGT2 = o3d.geometry.PointCloud()
    mGT2.points = o3d.utility.Vector3dVector(modGT)
    o3d.io.write_point_cloud("dati pc/modGT2.ply", mGT2)

    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = [defShape[:, 0]]
    y = [defShape[:, 1]]
    z = [defShape[:, 2]]

    ax.scatter3D(x,y,z, color='r')


    x1 = [modGT[:, 0]]
    y1 = [modGT[:, 1]]
    z1 = [modGT[:, 2]]

    ax.scatter3D(x1, y1, z1, color='b')
    plt.show()
    """
    # Initial Association
    [modPerm, err, minidx, missed] = bidirectionalAssociation(modGT, defShape)
    err_init = err

    # Re-align

    iidx = np.setdiff1d(np.arange(1, np.size(defShape, 0)), missed)
    [A, S, R, trasl] = _3DMM.estimate_pose(modGT[minidx[iidx], :], defShape[iidx, :])
    modPerm = np.transpose(_3DMM.getProjectedVertex(modPerm, S, R, trasl))
    modGT = np.transpose(_3DMM.getProjectedVertex(modGT, S, R, trasl))
    print('Mean distance initialization: ' + str(err))
    # ..........................................................

    # NRF ....................................
    print('Start NRF routine')
    d = 1
    t = 1
    alphas = []  # Keep for Recognition
    alphas.append([])
    while t < maxIter and d > derr:
        # Fit the 3dmm
        alpha = _3DMM.alphaEstimation_fast_3D(defShape, modPerm, Components_res, np.arange(0, 6704), weights, lambda_all)
        defShape = np.transpose(_3DMM.deform_3D_shape_fast(np.transpose(defShape), components, alpha))
        """
        ax = plt.axes(projection='3d')

        x = [defShape[:, 0]]
        y = [defShape[:, 1]]
        z = [defShape[:, 2]]

        ax.scatter3D(x, y, z, c=z, cmap='Greens')
        plt.show()
        """
        # Re-associate points as average
        [modPerm, errIter, minidx, missed] = bidirectionalAssociation(modGT, defShape)
        d = np.abs(err - errIter)
        err = errIter

        err_lm = np.mean(np.diag(cdist(modGT[lmidxGT_all, :], modPerm[lm3dmmGT_all.astype(int), :])))

        print('Mean distance: ' + str(err) + ' - Mean Landmark error - ' + str(err_lm))

        # Iterate
        t = t + 1
        #alphas = np.append(alphas, alpha, axis=1)
        #alphas = np.append(alphas, alpha, axis=1)
        #alphas = np.append(alphas, alpha, axis=1)
    # ...................
    """
    # Debug .............
    if debugMesh:
        plt.figure()
        plt.subplot(1, 2, 1)
        # plot_landMesh dove è implementata? eng.
        plt.title('NRF')
        plt.subplot(1, 2, 2)
        # plot_landMesh eng.
        plt.title('GT model')
        plt.pause()
        #eng.close
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
    missingModels = [missingModels, meshList[i]]

    print(f'Model {i} out of {len(meshList)}')
