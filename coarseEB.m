function coarseEB( image_dir )
    %% if candidates exist, return
    candidates = dir('candidates/EB/*.mat');
    if size(candidates, 1) ~= 0, return; end

    %% load pre-trained edge detection model and set opts (see edgesDemo.m)
    model=load('edges/models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

    %% set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .70;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e6;  % max number of boxes to detect

    %% detect Edge Box bounding box proposals (see edgeBoxes.m)
    images = dir(fullfile(image_dir, '*.png'));
    n = size(images, 1);
    parfor i = 1:n
        img = fullfile(image_dir, images(i).name);
        im = imread(img);
        bbs=edgeBoxesNoNMS(im,model,opts);
        parSave(sprintf(['candidates/EB/' images(i).name(1:6) '.mat']), bbs);
    end

end

