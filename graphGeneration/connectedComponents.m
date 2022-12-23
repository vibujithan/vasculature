function [sortedIdx,pixelList] = connectedComponents(I, connectivity)
% Compute connectedness between vessels based on the connectivity
% criteria and provide the list of voxel Ids in descending order

CC = bwconncomp(I,connectivity);
numPixels = cellfun(@numel,CC.PixelIdxList);

[~, sortedIdx] = sort(numPixels,'descend');
pixelList = CC.PixelIdxList;
