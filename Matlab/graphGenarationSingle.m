  
  addpath(genpath('.'));
    
  resolution = [0.93,0.93,0.93];
  binaryVolume = readBinaryVolume('./Data/Segmented/vasc02.tif',0,0);

  figure;
  volshow(binaryVolume);

  vG = generateVesselGraph(binaryVolume, resolution);
 
  figure;
  viewNetworkGraph(vG,resolution);

  sVG = simplifyVesselGraph(vG, resolution);
   
  figure;
  viewNetworkGraph(sVG,resolution);