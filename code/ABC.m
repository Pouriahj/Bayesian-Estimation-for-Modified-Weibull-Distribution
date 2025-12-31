%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Approximate Bayesian Compuatation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Author: Pouria Hajizadeh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Winter 2021 - December 31 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loading Data (Natural Fiber Info ...)

clc;
clear;
close all;

Data=xlsread('Date Palm Fiber strength data.xlsx');        % Importing our data from Excel    
L = Data(:, 1);                                                                     % Fiber Lengths
V = Data(:, 4);                                                                     % Fiber Volumes
Strength=Data(:, 5);                                                           % Ultimate sterngth of fibers

% In this section we distinguish every fiber strength with its related length

lendata = length(Data);
Lvalue = [10, 15, 20, 25, 30, 40, 50];                              % we know from our data that the fiber length consist of these 6 value
Llen = zeros(length(Lvalue), 2);
for i=1:length(Lvalue)
    temp = find(L==Lvalue(i));
    temp2 = temp([1, end]);
    Llen(i, :) = temp2';
end

%% Defining Parameters

T = 500000;                                   % Iteration 
rng('default');                                % for fixing the initial seed of the random process
Theta = zeros(3, T);                       % THETA matrix defined for the parameters generated in each interation 
decisions = zeros(3, T);                  % DECISION matrix defined for acceptance rate (it contains of "0" and "1" meaned that it is accepted or rejected)

% Determing an intervla for the parameters (sigma0, Beta, m)

theta1min = 0;                               
theta1max = 450;
theta2min = 0;
theta2max = 1.6;
theta3min = 0;
theta3max = 15;

% generating random value for parameters uniformly and without any bias

theta1 = unifrnd(theta1min, theta1max, 1, T);
theta2 = unifrnd(theta2min, theta2max, 1, T);
theta3 = unifrnd(theta3min, theta3max, 1, T);

%% Initiating the ABC Algorithm

t=0;                                               % temporary value used in WHILE loop
Freq3 = [];                                     % defining a null matrix for accepted parameters

while t<T
    t=t+1;
    
    Y = my_weibull_dist(V, lendata, theta2(t), theta1(t), theta3(t));
    for i=1:length(Lvalue)
        Y(Llen(i, 1):Llen(i, 2)) = sort(Y(Llen(i, 1):Llen(i, 2)));
    end
    norm_vec = norm(Y - Strength);
    if norm_vec<300
        Freq3 = [Freq3, [theta1(t); theta2(t); theta3(t)]];
    end    
end

toc;

%% Chi-Squared Goodness of Fit - Table 2

% calculating the chi-squared goodness of fit P-Value (p1, p2, p3) with C.I. of 95%

pd1 = fitdist(Freq3(1, :)','lognormal');
pd2 = fitdist(Freq3(2, :)','weibull');
pd3 = fitdist(Freq3(3, :)','lognormal');


[h1, p1, st1] = chi2gof(Freq3(1, :), 'cdf', pd1, 'alpha', 0.05, 'Emin', 5);
[h2, p2, st2] = chi2gof(Freq3(2, :), 'cdf', pd2, 'alpha', 0.05, 'Emin', 5);
[h3, p3, st3] = chi2gof(Freq3(3, :), 'cdf', pd3, 'alpha', 0.05, 'Emin', 5);


%% Acceptance Rate - Figure 7

% I run three algorithm in advance and these are the acceptance rate of
% each algorithm and it's as follow: 
% row1 ------> ABC   |   row2 ------> ABC SMC   |   row3 ------> ABC MCMC

y = [3.4e-3, 0.0368, 0.172; 3.3e-3, 0.0268, 0.179; 3.1e-3, 0.02115, 0.1852; 3.26e-3, 0.01932, 0.18558; 2.81e-3, 0.0192, 0.18744];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ploting Section %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = categorical({'N = 5K','N = 10K','N = 20K','N = 50K', 'N = 100K'});
X = reordercats(X, {'N = 5K','N = 10K','N = 20K','N = 50K', 'N = 100K'});

rgb1 = [0.4660 0.6740 0.1880];
rgb2 = [0.3010 0.7450 0.9330];	
rgb3 = [0.6350 0.0780 0.1840];

figure;
samp1 = bar(X, y, 'facecolor', 'flat', 'edgecolor', 'none');
ylabel('Acceptance Rate', 'fontsize', 15, 'fontname', 'times')
xlabel('Iteration', 'fontsize', 15, 'fontname', 'times')
set(gca, 'fontsize', 14, 'fontname', 'times')
