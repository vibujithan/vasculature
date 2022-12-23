function vG = vesselSegments(vG)
% vG.Edges.region = cell(height(vG.Edges),1);
vG.Edges.vol = zeros(height(vG.Edges),1);
vG.Edges.cx = zeros(height(vG.Edges),1);
vG.Edges.cy = zeros(height(vG.Edges),1);
vG.Edges.cz = zeros(height(vG.Edges),1);

nodeDegree = degree(vG);
nodesOfInterest = find(nodeDegree ==2);

toRemove = [];

while ~isempty(nodesOfInterest)
    
    stack_s = nodesOfInterest(1);
    line_seg = [];
    terms = [];
    
    while ~isempty(stack_s)
        vertex = stack_s(end);
        stack_s(end) = [];
        
        neigh = neighbors(vG,vertex);
        
        if length(neigh) == 2
            nodesOfInterest(nodesOfInterest == vertex) = [];
            line_seg = [line_seg vertex];
            stack_s = [stack_s setdiff(neigh,union(union(stack_s,line_seg),terms))'];
        else
            terms = [terms vertex];
        end
    end
    
    if length(terms) ==2
        
        sg = subgraph(vG, union(line_seg,terms));
        segRad = mean(sg.Edges.rad);
        segDist = sum(sg.Edges.length);
        
       
        vecs = table2array(vG.Nodes(terms,1:3));
        diff = vecs(2,:) - vecs(1,:);
        vec = diff./vecnorm(diff);
        
        segX = vec(1);
        segY = vec(2);
        segZ = vec(3);
        
        sg = subgraph(vG, line_seg);
        vol = sum(sg.Nodes.volume);
        cx = round(mean(sg.Nodes.x));
        cy = round(mean(sg.Nodes.y));
        cz = round(mean(sg.Nodes.z));        
        
        newEdge = table(terms,segX,segY,segZ,segRad,segDist,vol, cx,cy,cz,...
            'VariableNames', {'EndNodes','xO','yO','zO','rad','length','vol','cx','cy','cz'});
        
        vG = addedge(vG,newEdge);
        toRemove = [toRemove line_seg];        
    end
end

vG = rmnode(vG,toRemove);
vG = simplify(vG,'max','PickVariable','vol');
vG = rmnode(vG,find(degree(vG) == 0));