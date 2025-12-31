%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Approximate Bayesian Compuatation - Markov Chain Monte Carlo (ABC MCMC) %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Metropolis - Hasting as a Family of MCMC Sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Author: Pouria Hajizadeh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Spring 2022 - January 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

tic;

T = 10000;                                         % "T" exhibits the Iteration porcess
threshold = 300;                                % defining threshold value

lambda1 = 100;                                 % Lambda1 exhibits the 1st parameter of the gamma dist for "sigma0" ("sigma0" proposal dist)
lambda2 = 1;                                     % Lambda2 exhibits the 1st parameter of the gamma dist for "Beta" ("Beta" proposal dist) 
lambda3 = 5;                                     % Lambda3 exhibits the 1st parameter of the gamma dist for "m" ("m" proposal dist)                     

tau1 = 5;                                            % tau1 exhibits the 2nd parameter of the gamma dist for "sigma0" ('sigma0" proposal dist) 
tau2 = 5;                                            % tau2 exhibits the 2nd parameter of the gamma dist for "Beta" ('Beta" proposal dist) 
tau3 = 5;                                            % tau3 exhibits the 2nd parameter of the gamma dist for "m" ('m" proposal dist) 

rng('default');                                    % It is suggested to define the "initial seed" in randomness processes, here I used the default seed value of the matlab 

% We defining a THETA MATRIX for our parameter with "row" = Iteration process and "column" = parameters size(3)

Theta = zeros(3, T);                          

% For bring up that weather a parameter accpeted or not we define a DECISIONS MATRIX which the "num 1" show that it is accpeted and "num 0" shows that it is rejected
% NOTE: every row might has its own particular "0 and 1 arrays" arrangement 'cause it is used "component wise accpet - reject process"

decisions = zeros(3, T);                      

t=1;                                                     % define a temp value of "t" for using further in the loop process
theta1 = 150;                                      % defining the initial value of "sigma0" parameter
theta2 = 0.2;                                       % defining the initial value of "Beta" parameter
theta3 = 2;                                          % defining the initial value of "m" parameter

Theta(1, t) = theta1;                           % leave the inital value of "sigma0" in the matrix THETA
Theta(2, t) = theta2;                           % leave the inital value of "Beta" in the matrix THETA                  
Theta(3, t) = theta3;                           % leave the inital value "m" in the matrix THETA

decisions(1, t) = 1;                              % it is assumed that the initial vlaue is accpeted so we leave num 1 for 1st column of the DECISION MATRIX
decisions(2, t) = 1;
decisions(3, t) = 1;

%% Applying ABC MCMC (MH) Algorithm

% we define the null matrix of FREQ which used to leave the parameters that satisfy the ABC process 
% NOTE: The size of the FREQ matrix is gonna change by various paramaters defined above and iteration process, thus it has no specific size

Freq = [];

% a WHILE loop till the temporary value of "t" reaches iteration process "T"

while t<T
    
    t=t+1;
    
    % "sigma0" generated value via its proposal gamma dist
    % "theta1_star" stands for generated value of "sigma0" - NEW ONE
    
    theta1_star = gamrnd(theta1*tau1, 1/tau1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% JUDGEMENT PROCESS for NEW generated "sigma0" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1st ratio is the probability (PDF) of the OLD "sigma0" to NEW "sigma0"
    
    asymmetric_ratio = gampdf(theta1, theta1_star*tau1, 1/tau1)/gampdf(theta1_star, theta1*tau1, 1/tau1);   
    
    % 2nd ratio is the probability (PDF) of the prior dist of the NEW "sigma0" to OLD "sigma0"
    
    ratio1 = exppdf(theta1_star, lambda1)/exppdf(theta1, lambda1);
    
    % 3rd ratio is the probability (PDF) of the likelihood dist (Modified Weibull) of the NEW "sigma0" to OLD "sigma0"
    % NOTE1: ratio2 is vector
    % NOTE2: "mod_wblpdf" is user defined function
    
    ratio2 = mod_wblpdf(Strength, V, theta1_star, theta2, theta3)./mod_wblpdf(Strength, V, theta1, theta2, theta3);
    
    % Multiplying ratios
    
    posterior_ratio = prod(ratio2).*ratio1;
    
    % calculating the alpha value
    
    alpha = min(1, posterior_ratio*asymmetric_ratio);
    
    % choosing a random num between 0 and 1
    
    u =rand;
    
    % accpet - reject of "sigma0"
    
    if alpha>u
        theta1 = theta1_star;
        decisions(1, t) = 1;
    end
    
    % "Beta" generated value via its proposal gamma dist
    % "theta2_star" stands for generated value of "Beta" - NEW ONE
    
    theta2_star = gamrnd(theta2*tau2, 1/tau2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% JUDGEMENT PROCESS for NEW generated"BETA" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1st ratio is the probability (PDF) of the OLD "Beta" to NEW "Beta"
    
    asymmetric_ratio = gampdf(theta2, theta2_star*tau2, 1/tau2)/gampdf(theta2_star, theta2*tau2, 1/tau2);
    
    % 2nd ratio is the probability (PDF) of the prior dist of the NEW "Beta" to OLD "Beta"
    
    ratio1 = exppdf(theta2_star, lambda2)/exppdf(theta2, lambda2);
    
    % 3rd ratio is the probability (PDF) of the likelihood dist (Modified Weibull) of the NEW "Beta" to OLD "Beta"
        
    ratio2 = mod_wblpdf(Strength, V, theta1, theta2_star, theta3)./mod_wblpdf(Strength, V, theta1, theta2, theta3);
    
    % Multiplying ratios
    
    posterior_ratio = prod(ratio2).*ratio1;
    
    % calculating the alpha value
     
    alpha = min(1, posterior_ratio*asymmetric_ratio);
    
    % choosing a random num between 0 and 1
    
    u =rand;
    
    % calculating the alpha value
    
    if alpha>u
        theta2 = theta2_star;
        decisions(2, t) = 1;
    end
    
    % "m" generated value via its proposal gamma dist
    % "theta3_star" stands for generated value of "m" - NEW ONE
    
    theta3_star = gamrnd(theta3*tau3, 1/tau3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% JUDGEMENT PROCESS for NEW generated "m" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % 1st ratio is the probability (PDF) of the OLD "m" to NEW "m"
    
    asymmetric_ratio = gampdf(theta3, theta3_star*tau3, 1/tau3)/gampdf(theta3_star, theta3*tau3, 1/tau3);
    
    % 2nd ratio is the probability (PDF) of the prior dist of the NEW "m" to OLD "m"
    
    ratio1 = exppdf(theta3_star, lambda3)/exppdf(theta3, lambda3);
    
    % 3rd ratio is the probability (PDF) of the likelihood dist (Modified Weibull) of the NEW "m" to OLD "m"
    
    ratio2 = mod_wblpdf(Strength, V, theta1, theta2, theta3_star)./mod_wblpdf(Strength, V, theta1, theta2, theta3);
    
    % Multiplying ratios
    
    posterior_ratio = prod(ratio2).*ratio1;
    
    % calculating the alpha value
    
    alpha = min(1, posterior_ratio*asymmetric_ratio);
    
    % choosing a random num between 0 and 1
    
    u =rand;
    
    % accpet - reject of "m"
    
    if alpha>u
        theta3 = theta3_star;
        decisions(3, t) = 1;
    end
    
    % Leave the NEW value of the "sigma0", "Beta", "m" (weather it is accepted or rejected) in the THETA matrix
    
    Theta(1, t) = theta1;
    Theta(2, t) = theta2;
    Theta(3, t) = theta3;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ABC Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generating new strength value with parameters (Y) 
    % "my_webiull_dist" is a user-defined function  
    
    Y = my_weibull_dist(V, lendata, theta2, theta1, theta3);
    
    % sorting strength value for every specfic length and volume from small to large one 
    
    for i=1:length(Lvalue)
        Y(Llen(i, 1):Llen(i, 2)) = sort(Y(Llen(i, 1):Llen(i, 2)));
    end
    
    % calculating the norm of the differnece vector of the real strength (Strength) and generated strength (Y)
    % compare the norm of the differnece vector to threshold and accpe - reject it
    
    norm_vec = norm(Y - Strength);
    if norm_vec<threshold
        Freq = [Freq, [theta1; theta2; theta3]];
    end
end

toc;

%% Chi-Squared Goodness of Fit - Table 4

% calculating the chi-squared goodness of fit P-Value (p1, p2, p3) with C.I. of 95% 

pd1 = fitdist(Freq(1, :)','lognormal');
pd2 = fitdist(Freq(2, :)','normal');
pd3 = fitdist(Freq(3, :)','lognormal');

[h1, p1, st1] = chi2gof(Freq(1, :), 'cdf', pd1, 'alpha', 0.05, 'Emin', 5)
[h2, p2, st2] = chi2gof(Freq(2, :), 'cdf', pd2, 'alpha', 0.05, 'Emin', 5)
[h3, p3, st3] = chi2gof(Freq(3, :), 'cdf', pd3, 'alpha', 0.05, 'Emin', 5)

%%  Iteration Process Figure - Figure 5

% Visualizing the Interation process of the MCMC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "sigma0" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3, 1, 1)
plot(1:T, Theta(1, :), 'color', '#0213C2')
xlabel('No. of Iteration', 'fontsize', 12, 'fontname', 'times');
ylabel('\sigma_{0}', 'fontsize', 12, 'fontname', 'times');
H = gca;
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
set(gca, 'fontname', 'times', 'fontsize', 11)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "Beta" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3, 1, 2)
plot(1:T, Theta(2, :), 'color', '#0213C2')
ylim([0, 2]);
xlabel('No. of Iteration', 'fontsize', 12, 'fontname', 'times');
ylabel('\beta', 'fontsize', 12, 'fontname', 'times');
H = gca;
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
set(gca, 'fontname', 'times', 'fontsize', 11)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% "m" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3, 1, 3)
plot(1:T, Theta(3, :), 'color', '#0213C2')
ylim([0, 7]);
xlabel('No. of Iteration', 'fontsize', 12, 'fontname', 'times');
ylabel('{\it m}', 'fontsize', 12, 'fontname', 'times');
H = gca;
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
set(gca, 'fontname', 'times', 'fontsize', 11)
set(gcf, 'color', 'white')

%% Error Ellipses Figure - Figure 4 - Upper Section

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Ellipse of "sigma0" and "Beta"  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
scatter(Freq(1, :), Freq(2, :), 'filled', 'r', 'MarkerEdgeColor', "r");
hold on;
plot_ellipse(Freq(1, :), Freq(2, :));
H = gca;
xticks([205 265]);
yticks([-0.5, 1.6]);
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
axis([205, 265, -0.5, 1.6]);
xlabel('\sigma_{0}', 'fontsize', 26, 'fontname', 'times')
ylabel('\beta', 'fontsize', 26, 'fontname', 'times')
set(gca, 'fontname', 'times', 'fontsize', 25)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Ellipse of "sigma0" and "m"  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
scatter(Freq(1, :), Freq(3, :), 'filled', 'r', 'MarkerEdgeColor', "r");
hold on;
plot_ellipse(Freq(1, :), Freq(3, :));
H = gca;
xticks([205, 265]);
yticks([2.5, 9.5])
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
axis([205, 265, 2.5, 9.5]);
xlabel('\sigma_{0}', 'fontsize', 26, 'fontname', 'times')
ylabel('{\it m}', 'fontsize', 26, 'fontname', 'times')
set(gca, 'fontname', 'times', 'fontsize', 25)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Ellipse of "Beta" and "m"  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
scatter(Freq(2, :), Freq(3, :), 'filled', 'r', 'MarkerEdgeColor', "r");
hold on;
plot_ellipse(Freq(2, :), Freq(3, :));
H = gca;
xticks([-0.5, 1.6]);
yticks([2.5, 9.5]);
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
axis([-0.5, 1.6, 2.5, 9.5]);
xlabel('\beta', 'fontsize', 26, 'fontname', 'times')
ylabel('{\it m}', 'fontsize', 26, 'fontname', 'times')
set(gca, 'fontname', 'times', 'fontsize', 25)


