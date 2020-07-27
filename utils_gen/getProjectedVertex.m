function v = getProjectedVertex(vertex,S,R,t)
vertex = transpVertex(vertex);
rotPc = (R*vertex)';
v = (S*rotPc')+repmat(t,1,size(rotPc,1));
return