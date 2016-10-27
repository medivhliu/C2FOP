close all;
clear;
addpath(genpath(pwd));

%% Set Directories.
KITTI_DATASET = '/home/lxl/project/dataset/kitti/object/training';  %where image_2/training/ reference to.
image_dir = fullfile(KITTI_DATASET, 'image_2');


%% Stage 1, generate coarse candidates.
if ~exist('candidates/SS', 'dir'), mkdir('candidates/SS'); end
if ~exist('candidates/EB', 'dir'), mkdir('candidates/EB'); end
% Selective Search
coarseSS(image_dir);
% Edge Boxes
coarseEB(image_dir);
% compute extra features
computeExtraFeatures('candidates', 'SS');
computeExtraFeatures('candidates', 'EB');

%% Stage 2, refine coarse candidates.
gt_dir='KITTI/groundtruth';
dist_dir='KITTI/distance';
dist2road_dir='KITTI/dist2road';

category = 'Cyclist'; % category: 'Car', 'Pedestrian', or 'Cyclist'
train_set = 'train'; % training set: 'train' or 'trainval'
test_set= 'val'; % test set: 'val' or 'test'
level = 3; % difficulty level 1: easy, 2: moderate, 3: hard
coarse_sel = 'SS';

% train bayes model   
bayes = bayesModel(train_set, dist_dir, dist2road_dir, category, coarse_sel);
% get final proposals
C2FProposal(bayes, train_set, test_set, category, coarse_sel);
% evaluate
if strcmp(category, 'Car')
    threshold = 0.7;
else
    threshold = 0.5;
end

counts = 500;
cnts = [5000 2000 1000 500 200 100 50 20 10 5 2 1];
thrs = 0.5:0.05:1;
evaluate(test_set, level, category, coarse_sel, cnts, threshold);
evaluate(test_set, level, category, coarse_sel, counts, thrs);


