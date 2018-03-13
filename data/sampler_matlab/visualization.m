%close all;
clear all;

query = '000437';
list_file = ['/ssd/tmp/test_normal/', query];

font_size = 18;
fig_size = [8, 6];

f = fopen(list_file);
data = textscan(f, '%s %f');
shape_list = data{1};
fclose(f);

for i=1:1:length(shape_list)
    candidate = shape_list{i,1};
    obj_file = ['/ssd/dataset/SHREC2016/obj_txt/test_allinone/model_', candidate, '.obj'];

    [vertex, faces] = read_obj(obj_file);
    fig = figure('Visible', 'Off');
    trisurf(faces, vertex(:,1), vertex(:,2), vertex(:,3), ...
            'FaceColor', [0.8,0.8,0.8], 'EdgeColor', 'none', 'FaceLighting', 'flat', ...
            'AmbientStrength', 0.5, 'SpecularColorReflectance', 1);
    colormap(gray)
    light('Position',[-0.4 0.2 0.9], 'Style', 'infinite')

    lim_max = max(max([xlim;ylim;zlim]));
    lim_min = min(min([xlim;ylim;zlim]));
    xlim([lim_min, lim_max]);
    ylim([lim_min, lim_max]);
    zlim([lim_min, lim_max]);

    axis off
    set(fig, 'Units', 'Inches', 'Position', [0, 0, fig_size(1), fig_size(2)], 'PaperUnits', 'Inches', 'PaperSize', [fig_size(1), fig_size(2)]);
    set(gcf, 'PaperUnits', 'Inches');
    set(gcf, 'PaperPosition', [0, 0, fig_size(1), fig_size(2)]);
    saveas(fig, ['visualization/', query, '_', num2str(i), '_', candidate, '.png'], 'png');
    
    %% crop the white edges
    RemoveWhiteSpace([], 'file', ['visualization/', query, '_', num2str(i), '_', candidate, '.png']);
    
    if i>=6
        break;
    end
end

%close all;
clear all;


% 'FaceColor', [0.7,0.7,0.7]
% gouraud
