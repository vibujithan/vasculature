function [cropped,i1,j1,k1]  = cropROI(binaryVolume)
% Shrink bounding box only to foreground voxels, removing redundant
% background information.

blockSize = size(binaryVolume);

for i=1:blockSize(1)
    plane = squeeze(binaryVolume(i,:,:));
    sum_val = sum(plane(:));
    
    if sum_val
        i1 = i;
        break;
    end
end

for j=1:blockSize(1)
    i = blockSize(1)-j+1;
    
    plane = squeeze(binaryVolume(i,:,:));
    sum_val = sum(plane(:));
    
    if sum_val
        i2 = i;
        break;
    end
end

for i=1:blockSize(2)
    plane = squeeze(binaryVolume(:,i,:));
    sum_val = sum(plane(:));
    
    if sum_val
        j1 = i;
        break;
    end
end

for j=1:blockSize(2)
    i = blockSize(2)-j+1;
    plane = squeeze(binaryVolume(:,i,:));
    sum_val = sum(plane(:));
    
    if sum_val
        j2 = i;
        break;
    end
end


for i=1:blockSize(3)
    plane = squeeze(binaryVolume(:,:,i));
    sum_val = sum(plane(:));
    
    if sum_val
        k1 = i;
        break;
    end
end

for j=1:blockSize(3)
    i = blockSize(3)-j+1;
    plane = squeeze(binaryVolume(:,:,i));
    sum_val = sum(plane(:));
    
    if sum_val
        k2 = i;
        break;
    end
end

cropped = binaryVolume(i1:i2,j1:j2,k1:k2);