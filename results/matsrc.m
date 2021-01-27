mat = dir('*.mat'); 
scores = size(length(mat))
for q = 1:length(mat) 
    name = mat(q).name
    load(name);
    if(isempty(data1))
        continue
    end
    seq1 = ScanMatch_FixationToSequence(data1, ScanMatchInfo)
    seq2 = ScanMatch_FixationToSequence(data2, ScanMatchInfo)
    score = ScanMatch(seq1, seq2, ScanMatchInfo)
    scores(q) = score
    fprintf('score = %f\n', score);
end