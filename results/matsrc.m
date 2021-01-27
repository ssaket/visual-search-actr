%mat = dir('*.mat'); 
% scores = size(5000, 5000);
% dirinfo = dir('C:\Users\sktsa\Projects\visual-search-actr\results\salicon\');
% dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
% subdirinfo = cell(length(dirinfo));
% for K = 3 : length(dirinfo)
%   thisdir = dirinfo(K).name;
%   subdirinfo{K} = dir(fullfile(thisdir, '*.mat'));
%   for q = 1:length(subdirinfo{K}) 
%     name = strcat(thisdir, '\' , subdirinfo{K}(q).name);
%     load(name);
%     if(isempty(data1))
%         continue
%     end
%     seq1 = ScanMatch_FixationToSequence(data1, ScanMatchInfo);
%     seq2 = ScanMatch_FixationToSequence(data2, ScanMatchInfo);
%     score = ScanMatch(seq1, seq2, ScanMatchInfo);
%     scores(K,q) = score;
%     fprintf('score = %f\n', score);
%   end
% end
% tascores = nonzeros(scores);
% fprintf('total mean score = %f\n', tascores);

scores = size(1000);
mat = dir('C:\Users\sktsa\Projects\visual-search-actr\results\salicon\image_3\**\*.mat');
for q = 1:length(mat) 
    name = strcat('image_3', '\' ,mat(q).name);
    load(name);
    if(isempty(data1))
        continue
    end
    seq1 = ScanMatch_FixationToSequence(data1, ScanMatchInfo);
    seq2 = ScanMatch_FixationToSequence(data2, ScanMatchInfo);
    score = ScanMatch(seq1, seq2, ScanMatchInfo);
    scores(q) = score;
    fprintf('score = %f\n', score);
end