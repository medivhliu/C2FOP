function prior = computePrior(gt, ids, train_set, dist_dir, dist2road_dir, category, coarse_sel)
    prior_model_dir = ['models/prior/' coarse_sel];
    if (~exist(prior_model_dir, 'dir')), mkdir(prior_model_dir); end
    model_name = [prior_model_dir '/prior_' category '_' train_set '.mat'];
    
    % if model already exists, load and return
    if (exist(model_name, 'file')) 
        data = load(model_name); 
        prior = data.prior; 
        return; 
    end
    
    n = 1;
    area = []; diag = []; sd2 = []; dmd = []; dists = []; dist2road = [];
    for i=1:size(ids,1)
        if (mod(i, 100) == 0)
            disp(['compute prior:' num2str(i) '/ ' num2str(length(ids))]);
        end
        gt0 = gt{i};
        for j=1:size(gt0,1)
            if (gt0(j, 5) == 1), continue; end;  %ignored
            
            data = load(sprintf('%s/%s_distance.mat', dist_dir, ids{i}));
            depth = data.distance;
            data = load(sprintf('%s/%s_dist2road.mat', dist2road_dir, ids{i}));
            d2r = data.dist2road;
            
            x0 = round(gt0(j, 1) + gt0(j, 3) / 2);
            y0 = round(gt0(j, 2) + gt0(j, 4) / 2);
            
            dist = mean([depth(y0, x0) depth(y0-1, x0) depth(y0+1, x0) depth(y0 - 1, x0 -1) depth(y0, x0 - 1) ...
                    depth(y0 + 1, x0 - 1) depth(y0 + 1, x0 + 1) depth(y0, x0 + 1) depth(y0 - 1, x0 + 1)]);
            dist_road = mean([d2r(y0, x0) d2r(y0 - 1, x0) d2r(y0 + 1, x0) d2r(y0 - 1, x0 - 1) d2r(y0, x0 - 1) ...
                    d2r(y0 + 1, x0 - 1) d2r(y0 + 1, x0 + 1) d2r(y0, x0 + 1) d2r(y0-1, x0 + 1)]);
                
            if (dist > 100), continue; end;
            
            area(n) = gt0(j,3) .* gt0(j,4);
            diag(n) = sqrt(gt0(j,3) .^ 2 + gt0(j,4) .^ 2);
            sd2(n) = area(n) .* dist .^ 2;
            dmd(n) = diag(n) .* dist;
            dists(n) = dist;
            dist2road(n) = dist_road;
            n = n + 1;
        end
    end
    
    dist2road = dist2road(dist2road >= -0.5);
    [dist2road_mean, dist2road_sigma] = normfit(dist2road);

    sd2 = sd2(sd2 <= 10^7);
    [sd2_mean, sd2_sigma] = normfit(sd2);

    dmd = dmd(dmd <= 6000);
    [dmd_mean, dmd_sigma] = normfit(dmd);

    % save sampling prior
    prior.dist2road_mean = dist2road_mean;  
    prior.dist2road_sigma = dist2road_sigma;
    prior.sd2_mean = sd2_mean;
    prior.sd2_sigma = sd2_sigma;
    prior.dmd_mean = dmd_mean;
    prior.dmd_sigma = dmd_sigma;
    save(model_name, 'prior');
end