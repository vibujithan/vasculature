function cM = orientation(cM, blockSize,resolution)
% Identify edge orientation from connectivityMatrix cM.
%   Detailed explanation goes here

start = cM(:,1);
finish = cM(:,2);

[startX,startY,startZ] = ind2sub(blockSize,start);
[finishX,finishY,finishZ] = ind2sub(blockSize,finish);
clear start finish

xM = finishX-startX;
yM = finishY-startY;
zM = finishZ-startZ;

nrm = vecnorm([xM,yM,zM], 2, 2);

xV = xM./nrm;
yV = yM./nrm;
zV = zM./nrm;

cM(:,3) = 1;
cM(:,4) = xV;
cM(:,5) = yV;
cM(:,6) = zV;

d = sqrt(sum(([xM,yM,zM].*resolution).^2,2));
cM(:,8) = d;
