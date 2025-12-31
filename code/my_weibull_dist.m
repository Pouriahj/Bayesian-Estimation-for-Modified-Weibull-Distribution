function y = my_weibull_dist(V, n, Beta, sigma0, M)
% generating random variable from modified Weibull dist
% Input: V--> volume and somtime length, n--> number of value, --> "Beta", "sigma0" and "M" are the dist parameters
rnd = rand(n, 1);
y = nthroot(-((log(1-rnd)).*(sigma0^M))./(V.^Beta), M);
end
