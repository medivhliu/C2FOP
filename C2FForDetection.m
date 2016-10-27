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
test_set= 'trainval'; % test set: 'val', 'test' or 'trainval'
coarse_sel = 'EB';

% train bayes model   
bayes = bayesModel(train_set, dist_dir, dist2road_dir, category, coarse_sel);
% get final proposals
C2FProposalForDetection(bayes, train_set, test_set, category, coarse_sel);
