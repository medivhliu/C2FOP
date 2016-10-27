
function evaluate(test_set, level, category, coarse_sel, cnts, thrs)
    %get image ids
    img_list_dir = 'KITTI/imgId';
    img_list= [img_list_dir '/' test_set '.txt'];
    if(~exist(img_list, 'file')), error('ids file not found'); end
    f = fopen(img_list); 
    ids = textscan(f, '%s %*s');
    ids = ids{1};
    fclose(f);

    % load dt
    proposal_dir = ['proposal/' coarse_sel '/' category];
    dt = cell(size(ids, 1), 1);
    for j=1:size(ids, 1)
        bbs_file = [proposal_dir '/' ids{j}];
        data = load(bbs_file);
        bbs = data.bbs;
        dt{j}=bbs;
    end 
    % load gt
    gt_dir = ['KITTI/groundtruth/' category '/val_' num2str(level) '.mat'];
    data = load(gt_dir);
    gt = data.gt;
    
    for i = 1:length(cnts)
        % cut bbs
        for k = 1:size(ids, 1)
            bbs = dt{k};
            bbs = bbs(1:min(end,cnts(i)), :);
            dt{k} = bbs;
        end
        for j = 1:length(thrs)
            % if evaluation result exists, continne
            eval_dir = ['evaluate/' coarse_sel '/' category '/level_' num2str(level)];
            if ~exist(eval_dir, 'dir'), mkdir(eval_dir); end
            file_name = [eval_dir '/N' num2str(cnts(i)) '_T' int2str2(round(thrs(j)*100),2) '.txt']; 
            if(exist(file_name, 'file')), continue; end
            

            disp(['Evaluating ... Candidates = ' num2str(cnts(i)) ' Threshold = ' num2str(thrs(j))]);    
            [gt1, dt1] = bbGt('evalRes', gt, dt', thrs(j));
            [~, r] = bbGt('compRoc', gt1, dt1', 1);
            r = max(r);

            dlmwrite(file_name, r);
        end
    end
end


