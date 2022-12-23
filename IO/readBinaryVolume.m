function binaryVolume = readBinaryVolume(tifPath,start,finish)
% Read 3D binary volume from tifPath from start to finish.

% inputs,
%   tifPath      : Path of 3D image volume of segmented vessels. 
%   start        : Start reading slices from this value
%   finish       : Read slices upto this value
%                  Use 0 for the end
          
% outputs,
%   binaryVolume : logical valued 3D array
%  
% example,
%   binaryVolume = readBinaryVolume('./Data/Segmented/endo_block.tif',0,0)
%
% Written by Vibujithan.V, University of Auckland (2019)

info = imfinfo(tifPath);
availableSlices = size(info,1);

if start > availableSlices ||start <= 0, start = 1; end
if finish > availableSlices || finish <=0, finish = availableSlices; end

totalSlices = length(start:finish);
binaryVolume = zeros([info(1).Height,info(1).Width,totalSlices], 'logical');
fprintf('Reading image start : %d to finish: %d\n', start, finish);

parfor slice=1:totalSlices
    binaryVolume(:,:,slice) = imread(tifPath,start+slice-1);
end

