close all
clear
addpath(genpath(pwd));

index_file = 'KITTI/imgId/val.txt';
f=fopen(index_file);
ids=textscan(f,'%s %*s'); 
ids=ids{1}; 
fclose(f);


%Load selective search result

result_dir = 'candidates/SS';
dt_len = size(ids, 1);
dt = cell(dt_len, 1);
for i=1:dt_len
    boxes = load([result_dir '/' ids{i} '.mat']);
    boxes = boxes.bbs;
    dt(i) = {boxes};
end

disp('Dt load finished');

%Load ground truth
category = 'Car';
testset= 'val';
level = 1;    % difficulty level,0: all, 1: easy, 2: moderate, 3: hard
gt = load(['KITTI/groundtruth/' category '/val_' num2str(level) '.mat']);
gt = gt.gt;
disp('Gt load finished');



num_proposal = [10000 5000 2000 1000 500 200 100 50 20 10 5 2 1];
cnt = size(num_proposal, 2);
recall = zeros(cnt,1);
for i=1:size(num_proposal, 2)
    for j = 1:dt_len
        tmp = dt{j};
        dt(j) = {tmp(1:num_proposal(i),:)};
    end
    [gt1, dt1] = bbGt('evalRes', gt', dt', 0.7);
    [~, r] = bbGt('compRoc', gt1, dt1, 1);
    max_r = max(r);
    recall(i) = max_r;
    disp(max_r);
end
% save([root_dir '/selective_search_result/evaluation/' category '.txt' ], 'recall');
fid = fopen([root_dir '/ss_evaluation/' category '.txt'], 'w');
fprintf(fid, '%g\n', recall);
fclose(fid);
disp('done');






