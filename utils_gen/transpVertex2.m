function v=transpVertex2(vertex,n,dim)
v=vertex;
if size(vertex,n) ~= dim
    v=v';
end
end