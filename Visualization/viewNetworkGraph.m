function viewNetworkGraph(vG,resolution)

EdgeColor = [0.5 0.5 1];

h1 = axes;
h = plot(vG,'XData',-resolution(1)* vG.Nodes.y,'YData',-resolution(2)* vG.Nodes.x,'ZData',resolution(3)* vG.Nodes.z, ...
    'NodeLabel',{}, 'EdgeLabel',{}, 'LineWidth', 2, 'NodeColor', [0 0 0],...
    'EdgeColor',EdgeColor, 'MarkerSize',0.1);
axis equal;