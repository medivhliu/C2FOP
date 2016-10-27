function coarseSS(image_dir)
    %% if candidates exist, return
    candidates = dir('candidates/SS/*.mat');
    if size(candidates, 1) ~= 0, return; end
    
    %% parameters
    % Parameters. Note that this controls the number of hierarchical
    % segmentations which are combined.
    colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

    % Here you specify which similarity functions to use in merging
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};

    % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
    % Note that by default, we set minSize = k, and sigma = 0.8.
    ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
    sigma = 0.8;

    % After segmentation, filter out boxes which have a width/height smaller
    % than minBoxWidth (default = 20 pixels).
    minBoxWidth = 20;

    %% selective search
    images = dir(fullfile(image_dir, '*.png'));
    n = size(images, 1);

    parfor i = 1:n
        boxes = cell(length(colorTypes) * length(ks), 1);
        priority = cell(length(colorTypes) * length(ks), 1);
        % As an example, use a single image
        img = fullfile(image_dir, images(i).name);
        im = imread(img);

        idx = 1;
        for j=1:length(ks)
            k = ks(j); % Segmentation threshold k
            minSize = k; % We set minSize = k
            for c = 1:length(colorTypes)
                colorType = colorTypes{c};
                [boxes{idx, 1}, ~, ~, ~, priority{idx, 1}] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
                idx = idx + 1;
            end
        end
        boxes = cat(1, boxes{:}); % Concatenate boxes from all hierarchies
        priority = cat(1, priority{:}); % Concatenate priorities

        % Do pseudo random sorting as in paper
        priority = priority .* rand(size(priority));
        [priority, sortIds] = sort(priority, 'ascend');
        boxes = boxes(sortIds,:);
        bb = cat(1,[boxes priority]);
        bb = FilterBoxesWidth(bb, minBoxWidth);
        bb = BoxRemoveDuplicates(bb);
        bbs = [bb(:, 2) bb(:, 1) (bb(:, 4) - bb(:, 2) - 1) (bb(:, 3) - bb(:, 1) - 1) bb(:, 5)];
        parSave(sprintf(['candidates/SS/' images(i).name(1:6) '.mat']), bbs);
    end
end