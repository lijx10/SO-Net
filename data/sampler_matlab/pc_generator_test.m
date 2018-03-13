root = '/ssd/dataset/SHREC2016/obj_txt/test_allinone/';
N = 10000;
  
folder_content = dir(root);
folder_content = folder_content(3:end);

parfor j=1:1:length(folder_content)
    if folder_content(j).bytes > 100 && strcmp(folder_content(j).name(end-2:end), 'obj')
        obj_file = [root, folder_content(j).name];
        [pc, pc_normal] = sampler(obj_file, N);

        % write to txt
        dlmwrite([obj_file(1:end-3), 'txt'], [pc, pc_normal], 'delimiter', ' ');
    end
end

% scatter3(pc(:,1), pc(:,2), pc(:,3), 50, pc_normal, 'Marker', '.');

