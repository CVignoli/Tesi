function def_shape =  test_defComponents( avgModel, Components, comp, magnitude, random_comp )
alpha = zeros(size(Components,2),1);
if ~random_comp
    alpha(comp) = magnitude;
else
    alpha = rand(size(Components,2),1);
    alpha = rescale(alpha, -magnitude, magnitude);
    
end
def_shape = deform_3D_shape( avgModel', Components, alpha)';
end

