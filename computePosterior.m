function posterior = computePosterior(cues, datac, bayes)
    fpt = bayes.fpt;
    space = bayes.space;
    data0 = bayes.p0;
    data1 = bayes.p1;
    p_neg = bayes.p00;
    p_pos = bayes.p11;
    
    binNumber0 = cell(length(cues));
    binNumber1 = cell(length(cues));

    for cue_id = 1:length(cues)
        switch cues{cue_id}
            case 'ar'
                binNumber0{cue_id} = max(min(ceil((datac(:, 1) - fpt(1)) .* (1 / space(1))), 100), 1);
            case 'score'
                binNumber0{cue_id} = max(min(ceil((datac(:, 2) - fpt(3)) .* (1 / space(3))), 100), 1);
            case 'dpd'
                binNumber0{cue_id} = max(min(ceil((datac(:, 3) - fpt(5)) .* (1 / space(5))), 100), 1);
            case 'sd2'
                binNumber0{cue_id} = max(min(ceil((datac(:, 4) - fpt(7)) .* (1 / space(7))), 100), 1);
            case 'd2r'
                binNumber0{cue_id} = max(min(ceil((datac(:, 5) - fpt(9)) .* (1 / space(9))), 100), 1);
            otherwise
                display('error: cue name unknown');
        end
    end

    for cue_id = 1:length(cues)
        switch cues{cue_id}
            case 'ar'
                binNumber1{cue_id} = max(min(ceil((datac(:, 1) - fpt(2)) .* (1 / space(2))), 100), 1);
            case 'score'
                binNumber1{cue_id} = max(min(ceil((datac(:, 2) - fpt(4)) .* (1 / space(4))), 100), 1);
            case 'dpd'
                binNumber1{cue_id} = max(min(ceil((datac(:, 3) - fpt(6)) .* (1 / space(6))), 100), 1);
           case 'sd2'
                binNumber1{cue_id} = max(min(ceil((datac(:, 4) - fpt(8)) .* (1 / space(8))), 100), 1);
           case 'd2r'
                binNumber1{cue_id} = max(min(ceil((datac(:, 5) - fpt(10)) .* (1 / space(10))), 100), 1);
            otherwise
                display('error: cue name unknown');
        end
    end

    for i = 1:length(binNumber0{1})
        p01(i) = data0{1}(binNumber0{1}(i)) + eps;%AR
        p02(i) = data0{2}(binNumber0{2}(i)) + eps;%SC
        p03(i) = data0{3}(binNumber0{3}(i)) + eps;%DPD
        p04(i) = data0{4}(binNumber0{4}(i)) + eps;%SD2
        p05(i) = data0{5}(binNumber0{5}(i)) + eps;%D2R

        p11(i) = data1{1}(binNumber1{1}(i)) + eps;%AR
        p12(i) = data1{2}(binNumber1{2}(i)) + eps;%SC
        p13(i) = data1{3}(binNumber1{3}(i)) + eps;%DPD
        p14(i) = data1{4}(binNumber1{4}(i)) + eps;%SD2
        p15(i) = data1{5}(binNumber1{5}(i)) + eps;%D2R
    end
    posterior = (1000 .* p_pos .* p11 .* p13 .* p14 .* p15) ./ (p_neg .* p01 .* p03 .* p04 .* p05 + p_pos .* p11 .* p13 .* p14 .* p15);
end
