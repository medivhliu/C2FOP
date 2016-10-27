function computeExtraFeatures(candidates_dir, coarse_sel)
    bbs_dir = [candidates_dir '/' coarse_sel];
    result_dir = ['features/' coarse_sel];
    distance_dir = 'KITTI/distance';
    dist2road_dir = 'KITTI/dist2road';
    index_file = 'KITTI/imgId/trainval.txt';
    
    if ~exist(result_dir, 'dir'), mkdir(result_dir); end
    
    % if file exist, return directly
    files = dir([result_dir '/*.mat']);
    if size(files, 1) ~= 0, return; end

    f = fopen(index_file);
    ids = textscan(f,'%s %*s'); 
    ids = ids{1}; 
    fclose(f);

    %read boxes
    parfor i = 1:size(ids, 1)
        disp(i);
        data = load([bbs_dir '/' ids{i} '.mat']);
        bb = data.bbs;
        bbs = zeros(size(bb, 1), 7);
        bbs(:, 1:5) = bb(:, 1:5);
        data = load([distance_dir '/' ids{i} '_distance.mat']);
        distance_ = data.distance;
        data = load([dist2road_dir '/' ids{i} '_dist2road.mat']);
        dist2road = data.dist2road;
        x = round(bbs(:, 1) + bbs(:, 3) / 2 - 1);
        y = round(bbs(:, 2) + bbs(:, 4) / 2 - 1);
        for j = 1:size(bb, 1)
            x0 = x(j);
            y0 = y(j);
            bbs(j, 6) = mean([distance_(y0, x0) distance_(y0 - 1, x0) distance_(y0 + 1, x0) distance_(y0 - 1,x0 - 1) distance_(y0, x0 - 1) ...
                   distance_(y0 + 1, x0 - 1) distance_(y0 + 1, x0 + 1) distance_(y0, x0 + 1) distance_(y0 - 1, x0 + 1)]);
            bbs(j, 7) = mean([dist2road(y0, x0) dist2road(y0 - 1, x0) dist2road(y0 + 1, x0) dist2road(y0 - 1, x0 - 1) dist2road(y0, x0 - 1) ...
                   dist2road(y0 + 1, x0 - 1) dist2road(y0 + 1, x0 + 1) dist2road(y0, x0 + 1) dist2road(y0 - 1, x0 + 1)]);
        end

        parSave([result_dir '/' ids{i} '.mat'], bbs);
    end


end

