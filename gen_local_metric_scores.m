clear
addpath('metrics\')

%% Parameters
image_resize = 0.5;

FR_MET_BASEPATH = 'FR_met\';

% LIVE IQA
FR_MET_SUBPATH = 'LIVE\LIVE IQA DB\';
BASE_PATH = 'D:\DB\IQA\LIVE\LIVE IQA DB\';
info_file = 'LIVE_IQA.txt';
format = '%d %d %s %s %f %d %d';
ref_idx = 3;
dis_idx = 4;

% % TID2008
% FR_MET_SUBPATH = 'TID2008\';
% BASE_PATH = 'D:\DB\IQA\TID2008\';
% info_file = 'TID2008.txt';
% format = '%d %d %s %s %f';
% ref_idx = 3;
% dis_idx = 4;

% % TID2013
% FR_MET_SUBPATH = 'TID2013\';
% BASE_PATH = 'D:\DB\IQA\TID2013\';
% info_file = 'TID2013.txt';
% format = '%d %d %s %s %f';
% ref_idx = 3;
% dis_idx = 4;

%% Generate local quality metric score maps
fid = fopen(info_file);
info = textscan(fid, format, [inf, 7]);
fclose(fid);
nFiles = size(info{1},1);

put_path = [FR_MET_BASEPATH, FR_MET_SUBPATH]
fprintf('Save path: %s\n', put_path)
if ~exist(put_path, 'dir')
    mkdir(put_path);
end

for im_idx = 1:nFiles
    tic
    % Load info file: referece - distorted set
    if mod(im_idx, 50) == 0
        fprintf('%d/%d\n', im_idx, nFiles)
    end

    % Read reference image
    refFile = info{ref_idx}{im_idx};
    im_ref = imread([BASE_PATH, refFile]);
    im_ref = imresize(im_ref, image_resize);
    im_ref_gr = double(rgb2gray(im_ref));
    im_ref = double(im_ref);

    % Read distorted image
    disFile = info{dis_idx}{im_idx};
    im_dis = imread([BASE_PATH, disFile]);
    im_dis = imresize(im_dis, image_resize);
    im_dis_gr = double(rgb2gray(im_dis));
    im_dis = double(im_dis);

    % Test metrics
    [ssim, ssim_map] = ssim_index(im_ref_gr, im_dis_gr);

    % Store metric results
    [pathstr, name, ext] = fileparts(disFile);
    if ~exist([put_path, pathstr], 'dir')
        mkdir([put_path, pathstr]);
    end

    fid = fopen([put_path, disFile, '.ssim.bin'], 'wb');
    fwrite(fid, ssim_map, 'float32');
    fclose(fid);
end
fprintf('Finsihed: %f sec\n', toc)
