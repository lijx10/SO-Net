root = '/ssd/dataset/SHREC2016/train/';
N = 10000;

root_content = dir(root);
root_content = root_content(3:end);
folder_list = {};

for i=1:1:length(root_content)
    folder_name = root_content(i).name;
    folder_list{i} = folder_name;
    
    folder_content = dir([root, folder_name]);
    folder_content = folder_content(3:end);
    
    parfor j=1:1:length(folder_content)
        if folder_content(j).bytes > 100 && strcmp(folder_content(j).name(end-2:end), 'obj')
            obj_file = [root, folder_name, '/', folder_content(j).name];
            [pc, pc_normal] = sampler(obj_file, N);
            
            % write to txt
            dlmwrite([obj_file(1:end-3), 'txt'], [pc, pc_normal], 'delimiter', ' ');
        end
    end
    
    % scatter3(pc(:,1), pc(:,2), pc(:,3), 50, pc_normal, 'Marker', '.');
end

disp(folder_list)