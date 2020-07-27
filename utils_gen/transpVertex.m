function v=transpVertex(vertex)
v=vertex;
if size(vertex,1) ~=3
    v=v';
end
end