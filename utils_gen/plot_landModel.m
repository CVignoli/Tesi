function  plot_landModel( model, landModel, style, enum )

plot3(model(:,1),model(:,2),model(:,3),style);
hold on

if size(landModel,2) == 3
    if enum
        for l = 1:size(landModel,1)
            plot3(landModel(l,1),landModel(l,2),landModel(l,3),'bo')
            text(landModel(l,1),landModel(l,2),landModel(l,3),num2str(l),'Color','b')
        end
    else
        plot3(landModel(:,1),landModel(:,2),landModel(:,3),'bo')
    end
    
end

end

