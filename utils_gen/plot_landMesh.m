function  plot_landMesh( model, landModel, enum )

plot_mesh(model,compute_delaunay(model));
hold on

if size(landModel,2) == 3
    if enum
        for l = 1:size(landModel,1)
            plot3(landModel(l,1),landModel(l,2),landModel(l,3),'r.','Markersize',40)
            text(landModel(l,1),landModel(l,2),landModel(l,3),num2str(l),...
                'Color','g','FontSize',25)
        end
    else
        plot3(landModel(:,1),landModel(:,2),landModel(:,3),'r.','Markersize',40)
    end
    
end

end

