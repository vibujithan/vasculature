function cM = radiiEstimation(binaryVolume, cM, resolution)
% This function estimate radius at each node using rayburst algorithm
% Refer: "Rayburst sampling, an algorithm for automated three-dimensional
% shape analysis from laser scanning microscopy images  - Rodriguez et al."
%
% inputs,
%   binaryVolume : 3D image volume of segmented vessels. 
%                  Use logical datatype.
%   cM           : Connectivity matrix of the medial points
%   resolution   : Resolution of the input image in micro meters. 
%                  Use 3-element array, e.g. [0.9,0.9,1.2]
%   
% outputs,
%   cM           : Updated connectivity matrix with radius information
% 
%
% Written by Vibujithan.V, University of Auckland (2019)

N = 100;
outside = ~binaryVolume;
blockSize = size(binaryVolume);
l = size(cM,1);
radiiOfRays = zeros(l,N);

v = [1, 0, 0]';
n = [0,0,1]';
V = zeros(3,N);
Rz = @(t) [cos(t), -sin(t), 0; sin(t), cos(t), 0; 0, 0, 1];

i=0;
for t = 0:2*pi/N:2*pi-(2*pi/N)
    i=i+1;
    vR = Rz(t)*v;
    V(:,i) =vR;
end

% Mid point of the edge
[aX,aY,aZ] = ind2sub(blockSize, cM(:,1));
[bX,bY,bZ] = ind2sub(blockSize, cM(:,2));

pX  = round((aX+bX)./2);
pY  = round((aY+bY)./2);
pZ  = round((aZ+bZ)./2);

xM = bX-aX;
yM = bY-aY;
zM = bZ-aZ;

nrm = vecnorm([xM,yM,zM], 2, 2);
normDirections = [xM./nrm,yM./nrm,zM./nrm]';

% cast rays out from the plane normal to the edge orientation
parfor i=1:l    
    vD = normDirections(:,i);

    if(isequal(n,abs(vD)))
        VRotated = V;
    else
        VRotated = rotationMatrix(n,vD,0)*V;
    end
    
    VRotated = bsxfun(@rdivide,VRotated,sqrt(sum(VRotated.*VRotated))); % normalizing    
    radiiOfRays(i,:) = findEdgeRadius(outside,[pX(i);pY(i);pZ(i)],VRotated, resolution);
end

radiiOfRays(isinf(radiiOfRays)) = NaN;
radii = mean(radiiOfRays,2,'omitnan');
cM(:,7) = radii;

% fixing terminal points
if size(cM,1) > 1
    cM(1,7) = (cM(1,7) + cM(2,7))/2;
    cM(end,7) = (cM(end,7) + cM(end-1,7))/2;
end
end

%% Helper function to find how far each ray travelled before encountering a boundary

function radii = findEdgeRadius(outerSpace,middlePoint,direction, resolution)

N = size(direction,2);
blockSize =size(outerSpace);
radii = Inf(1,N);
limit = 100;

parfor d=1: N
    travel=0;
    foundEdge =false;
    while(travel < limit && ~foundEdge)
        travel=travel+1;
        cordinate = floor(middlePoint + travel.* [direction(1,d);direction(2,d);direction(3,d)]);
        x=cordinate(1);
        y=cordinate(2);
        z=cordinate(3);
        
        foundEdge = x <= 0 || y <= 0 || z <= 0 || x > blockSize(1) || y > blockSize(2) || z > blockSize(3)|| outerSpace(x,y,z);
        
        if(foundEdge)
              radii(d) = sqrt(sum((resolution' .* ((middlePoint - cordinate))).^2)); 
        end
    end     
end
end

%% Helper function to rotate the plane of rays normal to the edge orientation

function R = rotationMatrix(from,to,normalize)

if(normalize)
    a = from/norm(from);
    b = to/norm(to);
else
    a=from;
    b=to;
end

Gmat = @(A,B) [ dot(A,B) -norm(cross(A,B)) 0;
    norm(cross(A,B)) dot(A,B)  0;
    0              0           1];

Fmat = @(A,B) [ A (B-dot(A,B)*A)/norm(B-dot(A,B)*A) cross(B,A) ];

Umat = @(Fi,G) (Fi*G)/Fi;

R = Umat(Fmat(a,b),Gmat(a,b));

end