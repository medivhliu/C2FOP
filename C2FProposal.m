function C2FProposal(bayes, train_set, test_set, category, coarse_sel)
    proposal_dir = ['proposal/' coarse_sel '/' category];
    if(~exist(proposal_dir, 'dir')), mkdir(proposal_dir); end
    
    % if proposals exist, return
    files = dir([proposal_dir, '/*.mat']);
    if size(files, 1) ~= 0, return; end
    
    img_list_dir = 'KITTI/imgId';
    img_list= [img_list_dir '/' test_set '.txt']; 
    if(~exist(img_list, 'file')), error('ids file not found'); end
    f = fopen(img_list); 
    ids = textscan(f,'%s %*s'); 
    ids = ids{1}; 
    fclose(f);

    %load prior
    data = load(['models/prior/' coarse_sel '/prior_' category '_' train_set '.mat']);
    prior = data.prior;
    dist2road_mean = prior.dist2road_mean;
    dist2road_sigma = prior.dist2road_sigma;
    sd2_mean = prior.sd2_mean;
    sd2_sigma = prior.sd2_sigma;
    dpd_mean = prior.dmd_mean;
    dpd_sigma = prior.dmd_sigma;

    candidates_dir = ['features/' coarse_sel];

	parfor j = 1:size(ids, 1)
        candidates_file = [candidates_dir '/' ids{j}];
        data = load(candidates_file);
        bbs = data.bbs;

        if(mod(j, 100) == 0)
            disp(['Generating... ' num2str(j) '/' num2str(size(ids, 1))]);
        end
        ar = bbs(:, 3) ./ bbs(:, 4);
        diag = sqrt(bbs(:, 3) .^ 2 + bbs(:, 4) .^ 2);
        dist=bbs(:, 6); 
        dist2road = exp(-(bbs(:, 7) - dist2road_mean) .^ 2 ./ (2 * dist2road_sigma ^ 2));

        area = bbs(:, 3) .* bbs(:, 4);
        sd2 = area .* dist .^ 2;
        sd2 = exp(-(sd2 - sd2_mean) .^ 2 ./ (2 * sd2_sigma ^ 2));

        dpd = diag .* dist;
        dpd = exp(-(dpd - dpd_mean) .^2 ./ (2 * dpd_sigma ^ 2));

        score = bbs(:, 5);
        cues = {'ar', 'score', 'dpd', 'sd2', 'd2r'};
        datac = cat(2, ar, score, dpd, sd2, dist2road);
        posterior = computePosterior(cues, datac, bayes);

        bbs(:,8) = posterior';
        bbs(:,5)=bbs(:,8);
        [a, b] = sort(bbs(:, 5), 'descend');
        bbs = bbs(b, 1:5);
        scores = posterior(b)';
        bbs = bbs(1:10000, :);
        scores = scores(1:10000);
        bbs(:, 1:5) = [bbs(:, 1) bbs(:, 2) (bbs(:, 1) + bbs(:, 3) - 1) (bbs(:, 2) + bbs(:, 4) - 1) bbs(:, 5)];
        [bbs, ~] = boxesNMS(uint32(bbs(:, 1:4)), single(scores), 0.80);

        % save for evaluation
        bbs = [bbs(:, 1) bbs(:, 2) (bbs(:, 3) - bbs(:, 1) + 1) (bbs(:, 4) - bbs(:, 2) + 1) bbs(:, 5)];
        parSave([proposal_dir '/' ids{j} '.mat'], bbs);
	end 
end

