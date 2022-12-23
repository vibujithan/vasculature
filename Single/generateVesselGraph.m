function vG = generateVesselGraph(binaryVolume, resolution)
% This function generateVesselGraph will generate graph structure of a
% 3D binary volume of segmented vessels
%
% inputs,
%   binaryVolume : 3D image volume of segmented vessels. 
%                  Use logical datatype.
%   resolution   : Resolution of the input image in micro meters. 
%                  Use 3-element array, e.g. [0.9,0.9,1.2]
%   
% outputs,
%   vG           : Vessel graph - Undirected graph structure of the vessels.
%                  Nodes include index, coordinates, radius, and volume
%                  Edges include nodes, orientation, length, and radius
%
% example,
%   resolution = [1,1,1];
%   binaryVolume = readBinaryVolume('./Data/Segmented/endo_block.tif',0,0);
%   vG = generateVesselGraph(binaryVolume, resolution);% 
%   figure;
%   volshow(binaryVolume);
%   figure;
%   viewNetworkGraph(vG,resolution);
%
%
% Written by Vibujithan.V, University of Auckland (2019)

blockSize = size(binaryVolume); 
[sortedIdx,pixelList] = connectedComponents(binaryVolume,26);
threshold = 300; % remove isolated vessel components below this threshold

noOfCC = sum(cellfun(@length,pixelList(sortedIdx)) > threshold);
fprintf("No of components: %d\n", noOfCC);

cM_temps = cell(noOfCC,1);
mP_news = cell(noOfCC,1);
mR_news = cell(noOfCC,1);

for i=1:noOfCC
    
    CC = false(blockSize);
    CC(pixelList{sortedIdx(i)}) = true;
    
    % crop out to region of interest
    [cropped,i1,j1,k1] = cropROI(CC);
    
    % compute medial points and connectivity matrix
    [mP,mR,cM_temp] = findMedialPoints(cropped);
    
    % converting cropped ROI values to original block values
    if ~isempty(cM_temp)        
        [mPi,mPj, mPk] = ind2sub(size(cropped),mP);
        mP_new = sub2ind(blockSize,mPi+i1-1,mPj+j1-1,mPk+k1-1);
        
        [ri,rj,rk] = cellfun(@(x) ind2sub(size(cropped),x), mR,'UniformOutput', false);
        mR_new = cellfun(@(x,y,z) sub2ind(blockSize,x+i1-1,y+j1-1,z+k1-1), ri,rj,rk,'UniformOutput', false);
        
        [Pi,Pj, Pk] = ind2sub(size(cropped),cM_temp(:,1));
        cM_temp(:,1) = sub2ind(blockSize,Pi+i1-1,Pj+j1-1,Pk+k1-1);
        
        [Ci,Cj, Ck] = ind2sub(size(cropped),cM_temp(:,2));
        cM_temp(:,2) = sub2ind(blockSize,Ci+i1-1,Cj+j1-1,Ck+k1-1);
        
        mP_news(i) = {mP_new};
        cM_temps(i) = {cM_temp};
        mR_news(i) = {mR_new};        
    end
end

% Merge connected components values
cM = cat(1, cM_temps{:});
medialPoints = cat(2, mP_news{:});
medialRegions = cat(2, mR_news{:});

% Edge orientation
if ~isempty(cM)
    cM = orientation(cM,blockSize,resolution);
else
    vG = [];
    return
end

% Radius estimation at nodes
cM = radiiEstimation(binaryVolume, cM, resolution);

 %% Vessel graph generation
 
noOfConnections = size(cM,1);

parent = zeros(noOfConnections,1);
child = zeros(noOfConnections,1);

p = cM(:,1);
c = cM(:,2);

xO = cM(:,4);
yO = cM(:,5);
zO = cM(:,6);

rad = cM(:,7);
vesselLength = cM(:,8);

parfor i=1: noOfConnections
    nodeP = p(i);
    nodeC = c(i);
    parent(i,1) = find(medialPoints==nodeP,1);
    child(i,1) = find(medialPoints==nodeC,1);
end


vG = graph(parent,child,[],length(medialPoints));

[x,y,z] = ind2sub(blockSize,medialPoints);

vG.Nodes.x = x';
vG.Nodes.y = y';
vG.Nodes.z = z';
vG.Nodes.medialPoints = medialPoints';
vG.Nodes.volume = cellfun(@(x)length(x), medialRegions(1:vG.numnodes))';

vG.Edges.xO = xO;
vG.Edges.yO = yO;
vG.Edges.zO = zO;

vG.Edges.rad = rad;
vG.Edges.length = vesselLength;