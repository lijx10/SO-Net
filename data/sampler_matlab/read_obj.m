function [vertex, faces] = read_obj(file)

% get vertex and faces
text = fileread(file);
text = regexprep(text, '/[0-9]*', '');
data = textscan(text, '%c %f %f %f%*c');

type = data{1};
values = [data{2}, data{3}, data{4}];

v_idx = find(type=='v');
vertex = values(v_idx, :);

f_idx = find(type=='f');
faces = values(f_idx, :);

end