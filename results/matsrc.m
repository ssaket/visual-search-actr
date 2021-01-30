mat = dir('*.mat'); 
scores = size(5000, 5000);
dirinfo = dir();
dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
subdirinfo = cell(length(dirinfo));
for K = 3 : length(dirinfo)
  thisdir = dirinfo(K).name;
  subdirinfo{K} = dir(fullfile(thisdir, '*.mat'));
  for q = 1:length(subdirinfo{K}) 
    name = strcat(thisdir, '\' , subdirinfo{K}(q).name);
    load(name);
    if(isempty(data1))
        continue
    end
    seq1 = ScanMatch_FixationToSequence(data1, ScanMatchInfo);
    seq2 = ScanMatch_FixationToSequence(data2, ScanMatchInfo);
    score = ScanMatch(seq1, seq2, ScanMatchInfo);
    scores(K,q) = score;
    fprintf('score = %f\n', score);
  end
end
tascores = nonzeros(scores);
fprintf('total mean score = %f\n', mean(scores));