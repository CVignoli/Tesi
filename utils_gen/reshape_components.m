function [ Components_res ] = reshape_components( Components )
%RESHAPE_COMPONENTS reshpape the components vector. To be used with
%opt3DMM_fast
Components_res = [];
    for c=1:size(Components,2)
        comp = Components(:,c)';
        comp = reshape(comp',3,size(Components,1)/3)';        
        Components_res(:,:,c)=comp;
    end

end

