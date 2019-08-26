function [pc, pc_normal] = sampler(file, N)

%% get vertex and faces
[vertex, faces] = read_obj(file);

%% get triangle area and face normal
triangles_a = vertex(faces(:,1), :); % Fx3
triangles_b = vertex(faces(:,2), :); % Fx3
triangles_c = vertex(faces(:,3), :); % Fx3

a_b = triangles_b - triangles_a; % Fx3
a_c = triangles_c - triangles_a; % Fx3

a_bxa_c = cross(a_b, a_c, 2); % Fx3
tmp = sqrt(sum(a_bxa_c.^2, 2)); % Fx1
areas = 0.5 * tmp; % Fx1
normals = a_bxa_c ./ tmp;  % Fx3

%% weighted random sample
sampled_tri_idx = randsample(length(areas), N, true, areas);

sampled_a = triangles_a(sampled_tri_idx, :); % Nx3
sampled_b = triangles_b(sampled_tri_idx, :); % Nx3
sampled_c = triangles_c(sampled_tri_idx, :); % Nx3

pc_normal = normals(sampled_tri_idx, :); % Nx3

%% sample points
u = rand(N, 1); % Nx1
v = rand(N, 1);
invalid = u + v > 1
u(invalid) = 1 - u(invalid)
v(invalid) = 1 - v(invalid)

pc = sampled_a + u .* sampled_b + v .* sampled_c;

end

% toc;
% scatter3(pc(:,1), pc(:,2), pc(:,3), 50, pc_normal, 'Marker', '.')
