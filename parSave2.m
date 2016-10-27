function parSave2(file_name, boxes, scores)
    boxes3D = zeros(size(boxes, 1), 7);
    save(file_name, 'boxes', 'scores', 'boxes3D', '-v6');
end

