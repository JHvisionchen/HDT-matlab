
%error('Note: you need  to compile the Matconvnet according to Readme.txt, and then comment the FIRST line in run_HDT.m')

run('./matconvnet/matlab/vl_setupnn.m')

images_folder = '/media/cjh/datasets/tracking/OTB100/Basketball/img/';
pathAnno = '/media/cjh/datasets/tracking/OTB100/Basketball/groundtruth_rect.txt';
pathModel = '/media/cjh/cvpaper/git/models/imagenet-vgg-verydeep-19.mat';

show_visualization = 1;

images = dir(fullfile(images_folder,'*.jpg'));

len = size(images,1);
img_files = cell(len,1);
for i = 1:len
    img_files{i} = [images_folder images(i).name];
end

rect_anno = dlmread(pathAnno);
% rect_anno:  nx4 matrix, storing gt bounding box in the form of [left top width height]

init_rect = rect_anno(1,:);
target_sz = [init_rect(4), init_rect(3)];
pos = [init_rect(2), init_rect(1)] + floor(target_sz/2);

% extra area surrounding the target
padding = struct('generic', 2.2, 'large', 1, 'height', 0.4);

lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
interp_factor = 0.01;
cell_size = 4;
bSaveImage = 0;

[positions] = tracker_ensemble(img_files, pos, target_sz, ...
                                padding, lambda, output_sigma_factor, interp_factor, ...
                                cell_size, show_visualization, rect_anno, bSaveImage, pathModel);

% save results
rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
rects(:,3) = target_sz(2);
rects(:,4) = target_sz(1);
res.type = 'rect';
res.res = rects;

results=res;


