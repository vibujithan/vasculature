function viewNetworkGraph(vG,resolution)

EdgeColor = [0.5 0.5 1];

h1 = axes;
h = plot(vG,'XData',resolution(1)* vG.Nodes.x,'YData',resolution(2)* vG.Nodes.y,'ZData',resolution(3)* vG.Nodes.z, ...
    'NodeLabel',{}, 'EdgeLabel',{}, 'LineWidth', vG.Edges.rad, 'NodeColor', [0 0 1],...
    'EdgeColor',EdgeColor, 'MarkerSize',0.001);
set(h1, 'Zdir', 'reverse')
axis equal;