function bayes = bayesModel(train_set, dist_dir, dist2road_dir, category, coarse_sel)

    bayes_model_dir = ['models/bayes/' coarse_sel];
    if(~exist(bayes_model_dir,'dir')), mkdir(bayes_model_dir); end

    % if model already exists, load and return
    model_name = [bayes_model_dir '/bayes_' category '_' train_set '.mat'];
    if(exist(model_name,'file'))
        data = load(model_name); 
        bayes = data.bayes; 
        return; 
    end

    img_list_dir='KITTI/imgId';
    img_list= [img_list_dir '/' train_set '.txt']; 
    % get list of image ids
    if(~exist(img_list,'file')), error('ids file not found'); end
    f = fopen(img_list); 
    ids = textscan(f,'%s %*s'); 
    ids = ids{1}; 
    fclose(f);

    % groundtruth
    data = load(['KITTI/groundtruth/' category '/' train_set '.mat']);
    gt = data.gt;

    prior = computePrior(gt, ids, train_set, dist_dir, dist2road_dir, category, coarse_sel);
    dist2road_mean = prior.dist2road_mean;
    dist2road_sigma = prior.dist2road_sigma;
    sd2_mean = prior.sd2_mean;
    sd2_sigma = prior.sd2_sigma;
    dpd_mean = prior.dmd_mean;
    dpd_sigma = prior.dmd_sigma;

    candidates_dir = ['features/' coarse_sel];
    m0=0;m1=0;k0=0;k1=0;ar0=[];score0=[];dpd0=[];sd20=[];ar1=[];score1=[];dpd1=[];sd21=[];n=1;

    for i=1:size(ids,1)
        if (mod(i, 100) == 0)
            disp(['compute Bayes:' num2str(i) '/' num2str(length(ids))]);
        end
        candidates_file = sprintf('%s/%s.mat', candidates_dir, ids{i});
        data = load(candidates_file);
        bbs = data.bbs;
        boxes = double(bbs);

        if (all(gt{i}(:,5))), continue; end; 

        [oa, ~, boxes] = boxclass(gt{i}, boxes, 0.6, 1);
        if ~isempty(oa) 
            for j = 1:size(oa, 2)
                if(gt{i}(j, 5) == 1)
                    boxes(find(oa(:, j) >= 0.35), 8) = -1;   
                else
                    boxes(find(oa(:, j) >= 0.1 & oa(:, j) < 0.6), 8) = -1;
                end
            end
        end
        bbs0 = boxes(find(boxes(:, 8) == 0), :);
        L0 = length(bbs0);
        bbs0 = bbs0(randperm(L0, min(600, L0)), :);
        k0 = size(bbs0, 1);

        ar0(1 + m0:k0 + m0) = bbs0(:, 3) ./ bbs0(:, 4);
        score0(1 + m0:k0 + m0) = bbs0(:, 5);

        dpd_n = sqrt(bbs0(:, 3) .^2 + bbs0(:, 4) .^ 2) .* bbs0(:, 6);
        dpd0(1 + m0:k0 + m0) = exp(-(dpd_n - dpd_mean) .^ 2 ./ (2 * dpd_sigma ^ 2));

        sd2_n = bbs0(:, 3) .* bbs0(:, 4) .* bbs0(:, 6) .^ 2;
        sd20(1 + m0:k0 + m0) = exp(-(sd2_n - sd2_mean) .^2 ./ (2 * sd2_sigma ^ 2));

        d2r_n = bbs0(:, 7);
        d2r0(1 + m0:k0 + m0)= exp(-(d2r_n - dist2road_mean) .^2 ./ (2 * dist2road_sigma ^ 2));
        
        m0 = m0 + k0;

        bbs1 = boxes(find(boxes(:, 8) == 1), :);
        if ~isempty(bbs1)
            k1 = size(bbs1, 1);

            ar1(1 + m1:k1 + m1)=bbs1(:, 3) ./ bbs1(:, 4);
            score1(1 + m1:k1 + m1) = bbs1(:, 5);

            dpd_p = sqrt(bbs1(:, 3) .^ 2 + bbs1(:, 4) .^ 2) .* bbs1(:, 6);
            dpd1(1 + m1:k1 + m1) = exp(-(dpd_p - dpd_mean) .^ 2 ./ (2 * dpd_sigma ^ 2));

            sd2_p = bbs1(:, 3) .* bbs1(:, 4) .* bbs1(:, 6) .^ 2;
            sd21(1 + m1:k1 + m1) = exp(-(sd2_p - sd2_mean) .^2 ./ (2 * sd2_sigma ^ 2));

            d2r_p = bbs1(:, 7);
            d2r1(1 + m1:k1 + m1) = exp(-(d2r_p - dist2road_mean) .^2 ./ (2 * dist2road_sigma ^ 2));
            m1 = m1 + k1;
        end
    end

    [ar0_p, ar0_c] = hist(ar0, 100);
    [score0_p, score0_c] = hist(score0, 100);
    [dpd0_p, dpd0_c] = hist(dpd0, 100);
    [sd20_p, sd20_c] = hist(sd20, 100);
    [d2r0_p, d2r0_c] = hist(d2r0, 100);
    
    space_ar0 = (max(ar0_c) - min(ar0_c)) / (length(ar0_c) - 1);
    fpt_ar0 = min(ar0_c) - (space_ar0 / 2);
    space_score0 = (max(score0_c) - min(score0_c)) / (length(score0_c) - 1);
    fpt_score0 = min(score0_c) - (space_score0 / 2);
    space_dpd0 = (max(dpd0_c) - min(dpd0_c)) / (length(dpd0_c) - 1);
    fpt_dpd0 = min(dpd0_c) - (space_dpd0 / 2);
    space_sd20 = (max(sd20_c) - min(sd20_c)) / (length(sd20_c) - 1);
    fpt_sd20 = min(sd20_c) - (space_sd20 / 2);
    space_d2r0 = (max(d2r0_c) - min(d2r0_c)) / (length(d2r0_c) - 1);
    fpt_d2r0 = min(d2r0_c) - (space_d2r0 / 2);

    ar0_p = ar0_p ./ length(ar0);
    score0_p = score0_p ./ length(score0);
    dpd0_p = dpd0_p ./ length(dpd0);
    sd20_p = sd20_p ./ length(sd20);
    d2r0_p = d2r0_p ./ length(d2r0);


    [ar1_p, ar1_c] = hist(ar1, 100);
    [score1_p, score1_c] = hist(score1, 100);
    [dpd1_p, dpd1_c] = hist(dpd1, 100);
    [sd21_p, sd21_c] = hist(sd21, 100);
    [d2r1_p, d2r1_c] = hist(d2r1, 100);

    space_ar1 = (max(ar1_c) - min(ar1_c)) / (length(ar1_c) - 1);
    fpt_ar1 = min(ar1_c) - (space_ar1 / 2);
    space_score1 = (max(score1_c) - min(score1_c)) / (length(score1_c) - 1);
    fpt_score1 = min(score1_c) - (space_score1 / 2);
    space_dpd1 = (max(dpd1_c) - min(dpd1_c)) / (length(dpd1_c) - 1);
    fpt_dpd1 = min(dpd1_c) - (space_dpd1 / 2);
    space_sd21 = (max(sd21_c) - min(sd21_c)) / (length(sd21_c) - 1);
    fpt_sd21 = min(sd21_c) - (space_sd21 / 2);
    space_d2r1 = (max(d2r1_c) - min(d2r1_c)) / (length(d2r1_c) - 1);
    fpt_d2r1 = min(d2r1_c) - (space_d2r1 / 2);

    ar1_p = ar1_p ./ length(ar1);
    score1_p = score1_p ./ length(score1);
    dpd1_p = dpd1_p ./ length(dpd1);
    sd21_p = sd21_p ./ length(sd21);
    d2r1_p = d2r1_p ./ length(d2r1);

    space = cat(2, space_ar0, space_ar1, space_score0, space_score1, space_dpd0, space_dpd1, space_sd20, space_sd21, space_d2r0, space_d2r1);
    fpt = cat(2, fpt_ar0, fpt_ar1, fpt_score0, fpt_score1, fpt_dpd0, fpt_dpd1, fpt_sd20, fpt_sd21, fpt_d2r0, fpt_d2r1);
    p0 = {ar0_p', score0_p', dpd0_p', sd20_p', d2r0_p'};
    p1 = {ar1_p', score1_p', dpd1_p', sd21_p', d2r1_p'};
    n0 = size(find(ar0 > 0), 2);
    n1 = size(find(ar1 > 0), 2);
    p00 = n0 / (n0 + n1);
    p11 = n1 / (n0 + n1);
    bayes = struct('space', space, 'fpt', fpt, 'p0', {p0}, 'p1', {p1}, 'p00', p00, 'p11', p11, 'sd2', sd21);

    save(model_name, 'bayes');

end


