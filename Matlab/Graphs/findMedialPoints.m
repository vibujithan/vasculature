function [medialPoints,medialRegions,connectivityMatrix] = findMedialPoints(CC)
% This function identify medial points of a connected component CC, using
% a modified version of voxel-coding technique of Zhou & Toga
% [doi: 10.1109/2945.795212]
%
% inputs,
%   CC : A fully connected binary component
%
% outputs,
%   medialPoints        : identified medial points of the component
%   medialRegions       : Voxel regions associated with the medial points
%   connectivityMatrix  : connectivity information of the medial points
%
%
% Written by Vibujithan.V, University of Auckland (2019)


ssMeasure = 'quasi-euclidean';
groupNum = 5;

blockSize = size(CC);

%% Single Seeded Mask

% Setting random initial seed point 
foreGround = find(CC > 0);
seedId = foreGround(randi(length(foreGround)));
mask = false(blockSize);
mask(seedId)= true;

% Setting SeedIds as the maximum indexes of the initial seeding
ssCode = bwdistgeodesic(CC,mask,ssMeasure);
ssCode(isinf(ssCode)|isnan(ssCode))=0;
[~, seedIds] = max(ssCode(:));
mask = false(blockSize);
mask(seedIds)= true;

% Recomputing SSCode from the new seeds
ssCode = bwdistgeodesic(CC,mask,ssMeasure);
ssCode(isinf(ssCode)|isnan(ssCode))=0;

% Grouping computed SScodes

ssCodeGrouped = uint16(round(ssCode/groupNum))+1;
ssCodeGrouped(ssCode==0)=0;
ssCodeGrouped(seedId) = 1;

%% Compute medial points from each grouped SSCode regions

uvals = unique(ssCodeGrouped);
uvals(1)=[];

medialPoints =[];
medialRegions ={};
ssRegions = struct('ssValue',{},'regions', {});
uvalSize =length(uvals);

for K = 1 : uvalSize
    regionProperties = regionprops(ssCodeGrouped == uvals(K),'PixelIdxList','Centroid');
    ssRegions(K).ssValue =K;
    regions = [];

    for cR = 1: length(regionProperties)
        voxels = regionProperties(cR).PixelIdxList;
        centroid = regionProperties(cR).Centroid;
        try
            [x,y,z] = ind2sub(blockSize,voxels);
            k = dsearchn([x,y,z],[centroid(2),centroid(1),centroid(3)]);
            medialPoint = voxels(k);
        catch e
            disp(e.message)
            medialPoint = voxels(end);
        end
        medialPoints =[medialPoints, medialPoint];
        medialRegions = [medialRegions,voxels];
        currentRegion = struct('voxels',voxels,'medialPoint',medialPoint,'child',[]...
            ,'connectedChild',false);
        regions = [regions,currentRegion];
    end
    ssRegions(K).regions =regions;
end

%% Find connected regions and create connectivity matrix

connectivity =struct('conMat',{});
connectivityMatrix = [];

ssRegionsChild = ssRegions;
ssRegionsChild(1) =[];

k_length = length(ssRegions)-1;

parfor K = 1 : k_length
    parentRegions = ssRegions(K).regions;
    childRegions = ssRegionsChild(K).regions;
    parentMatrix =[];
    for pR = 1: length(parentRegions)
        parentVoxels = parentRegions(pR).voxels;
        medialPoint = parentRegions(pR).medialPoint;
        childMatrix =[];
        for cR =1:length(childRegions)
            childVoxels = childRegions(cR).voxels;
            if isConnected(parentVoxels,childVoxels,ssCode)
                childMedial  =childRegions(cR).medialPoint;
                childMatrix = cat(1, childMatrix, [medialPoint, childMedial, 0]);
            end
        end
        parentMatrix = cat(1, parentMatrix, childMatrix);
    end
    connectivity(K).conMat = parentMatrix;
end

for i=1:length(connectivity)
    connectivityMatrix = cat(1, connectivityMatrix, connectivity(i).conMat);
end
end

%% Helper function to define connectivity of regions

function connected = isConnected(region1,region2,ssCode)

maxR1 = max(ssCode(region1));
minR2 = min(ssCode(region2));

maxBoundary = region1(ssCode(region1)>= floor(maxR1));
minBoundary = region2(ssCode(region2)<= ceil(minR2));
connected =false;

[px1,py1,pz1] =ind2sub(size(ssCode),maxBoundary);
[px2,py2,pz2] =ind2sub(size(ssCode),minBoundary);

p1 = [px1,py1,pz1];
p2 = [px2,py2,pz2];

minDist = min(pdist2(p1,p2,'euclidean','Smallest',1));

if(minDist < 2)
    connected = true;
end
end
