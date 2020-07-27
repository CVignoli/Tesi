function show3DFaceHeatMap(avgFace,defShape)

diff = sqrt(sum((avgFace-defShape).^2,2));
%diff  = (diff - min(diff))/(max(diff) - min(diff));
options.face_vertex_color = diff;
plot_mesh(defShape, compute_delaunay(defShape),options);
colormap jet
colorbar
%caxis([0 3])
return