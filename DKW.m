function [low_edge, F_hat, hi_edge, x] = DKW(data, alpha)
    [F_hat, x] = ecdf(data);
    epsilon = sqrt(log(2/alpha)/(2*length(data))); 
    low_edge = max(F_hat - epsilon, 0); %Does the right thing here, use pmax in R
    hi_edge = min(F_hat + epsilon, 1);
end