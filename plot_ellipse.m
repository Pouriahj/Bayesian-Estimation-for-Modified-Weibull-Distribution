function h = plot_ellipse(vec1, vec2)
cov_mat = cov(vec1, vec2);
t = linspace(0, 2*pi, 500);
[U, V] = eig(cov_mat);
alpha = atan(U(2, 2)/U(1, 2));
minor_radii = sqrt(5.991*V(1));
major_radii = sqrt(5.991*V(4));
x = (major_radii.*cos(t)*cos(alpha))-(minor_radii*sin(t)*sin(alpha))+mean(vec1);
y = (minor_radii.*sin(t)*cos(alpha))+(major_radii*cos(t)*sin(alpha))+mean(vec2);
h = plot(x, y, "color", '#5D1CCF', "linewidth", 3);
end