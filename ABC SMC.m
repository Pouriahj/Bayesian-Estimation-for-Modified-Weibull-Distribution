%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Approximate Bayesian Compuatation - Sequential Monte Carlo (ABC SMC) %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SMC with Uniform Kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Author: Pouria Hajizadeh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Winter 2021 - December 31 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Loading Data (Natural Fiber Info ...)

clc;
clear;
close all;

Data=xlsread('Date Palm Fiber strength data.xlsx');       % Importing data from Excel    
L = Data(:, 1);                                                                    % Fiber Lengths
V = Data(:, 4);                                                                    % Fiber Volumes
Strength=Data(:, 5);                                                          % Ultimate sterngth of fibers 
NoP=3;                                                                              % Number of Parameters

% In this section we distinguish every fiber strength with its related length

lendata = length(Data);
Lvalue = [10, 15, 20, 25, 30, 40, 50];
Llen = zeros(length(Lvalue), 2);
for i=1:length(Lvalue)
    temp = find(L==Lvalue(i));
    temp2 = temp([1, end]);
    Llen(i, :) = temp2';
end

%% Defining Parameters

tic;

T = 10000;                                                                                                                                            % "T" exhibits the Iteration porcess
epsilonpop=[800, 750, 700, 650, 600, 550, 500, 450, 400, 380, 350, 300];                                        % epsilionpop is the sequence of thresholds - 12 sequence used here

lambda1 = 40;                                   % Lambda1 exhibits the only parameter of the exponential dist for "sigma0" ("sigma0" proposal dist)
lambda2 = 1;                                     % Lambda2 exhibits the only parameter of the exponential dist for "Beta" ("Beta" proposal dist) 
lambda3 = 1.5;                                  % Lambda3 exhibits the only parameter of the exponential dist for "m" ("m" proposal dist)                     

rng('default');                                    % It is suggested to define the "initial seed" in randomness processes, here I used the default seed value of the matlab

acc_rate = [];                                     % accaptance rate vector that claculate the number of accapted paramteres at the end of the sequence 12

% Here we define WEIGHT MATRIX as a " 6*iteration " so that we can leave
% old value (Previous sequence) and new value (Current sequence) of the
% parameters weight in the matrix

weight_matrix = zeros(6, T);

% THETA MATRIX is the matrix of the paramters of old value (previous sequence) and new value (current sequence)

Theta = zeros(6, T);

%% Applying the ABC SMC Algorithm

% FREQ2 is the matrix of accpeted parameters from the ABC judgement

Freq2 = [];
H = gca;

% Initiating the ABC SMC porcess

for ii=1:length(epsilonpop)
    dec_val = [];                                               % dec_val is the decision vector which provide weather the total parameters in ABC process is accepted or not
    t=0;                                                             % "t" is temporary value in each sequence and is let to be zero 
%     arr_tot = [];                                                
    while t<T
        t=t+1;
        if ii==1                                                                   % In sequence 1 we should generate parameters from the prior dist (exponential dist)
            theta1 = exprnd(lambda1);                               % generate "sigma0"
            theta2 = exprnd(lambda2);                               % generate "Beta"
            theta3 = exprnd(lambda3);                               % generate "m"
        else
            arr_theta1 = randsample(1:b, 1, true, weight_matrix_acc(1, :));         % generate a random value of "sigma0" from the previous accpeted parameter and its weight
            arr_theta2 = randsample(1:b, 1, true, weight_matrix_acc(2, :));         % generate random value of "Beta" from the previous accpeted parameter and its weight
            arr_theta3 = randsample(1:b, 1, true, weight_matrix_acc(3, :));         % generate random value of "m" from the previous accpeted parameter and its weight
            
            % leave the new value of parameters in THETA matrix
            
            Theta(4:6, t) = [Freq2(1, arr_theta1); Freq2(2, arr_theta2); Freq2(3, arr_theta3)];
            
            % leave the weight of the new value of parameters in WEIGHT matrix
            
            weight_matrix(4:6, t) = [weight_matrix_acc(1, arr_theta1); weight_matrix_acc(2, arr_theta2); weight_matrix_acc(3, arr_theta3)];
            
            % perturb the generated paramters corresponding to its Std value
            
            theta1 = unifrnd(Freq2(1, arr_theta1)-Std(1), Freq2(1, arr_theta1)+Std(1));
            theta2 = unifrnd(Freq2(2, arr_theta2)-Std(2), Freq2(2, arr_theta2)+Std(2));
            theta3 = unifrnd(Freq2(3, arr_theta3)-Std(3), Freq2(3, arr_theta3)+Std(3));
        end
        
        % leave the generated parameters (theta1, theta2, theta3) in the THETA matrix
        
        Theta(1, t) = theta1;                                               
        Theta(2, t) = theta2; 
        Theta(3, t) = theta3;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ABC Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % I lay down this condition 'cause in sequence>=2 it might appear negative value for paramters
        % generating new strength value with parameters (Y) 
        % "my_webiull_dist" is a user-defined function  
        
        if theta1>0 && theta2>0 && theta3>0
            Y = my_weibull_dist(V, lendata, theta2, theta1, theta3);
            
            % sorting strength value for every specfic length and volume from small to large one
            
            for i=1:length(Lvalue)
                Y(Llen(i, 1):Llen(i, 2)) = sort(Y(Llen(i, 1):Llen(i, 2)));
            end
            
            % calculating the norm of the differnece vector of the real strength (Strength) and generated strength (Y)
            % leave the accepted step that is accepted in DECISION vector (dec_val)
            
            norm_vec = norm(Y - Strength);
            if norm_vec<epsilonpop(ii)
                Freq2 = [Freq2, [theta1; theta2; theta3]];
                dec_val = [dec_val, t];
            end
        end
    end
       
    acc_rate = [acc_rate, length(dec_val)/T];        % calculating the acceptance rate in each sequence and add it up to the vector
   
    if ii~=1
        Freq2(:, 1:b) = [];                                             % for sequence that is not equal to 1, we need to clear away the Freq matrix
    end
    
    b = length(Freq2);                                                % 'cause we use the length of the Freq a lot we define the "b" parameter
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculating the Paramters Weight %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculating the weight for every value of the three parameters

    if ii==2 || ii==3 || ii==4 || ii==5 || ii==6 || ii==7 || ii==8 || ii==9 || ii==10 || ii==11 || ii==12
        kernel_const = prod(1./(2.*Std)); 
        for j=1:T 
            weight_matrix(1, j) = exppdf(Theta(1, j), lambda1)./(sum(kernel_const.*weight_matrix(4, :)));
            weight_matrix(2, j) = exppdf(Theta(2, j), lambda2)./(sum(kernel_const.*weight_matrix(5, :)));
            weight_matrix(3, j) = exppdf(Theta(3, j), lambda3)./(sum(kernel_const.*weight_matrix(6, :)));
        end
    elseif ii==1
        weight_matrix(1:3, :) = ones(3, T);
    end
       
    % we only pick up the weight of the parameters that is accpeted with the help of DECISION vector
    
    weight_matrix_acc = weight_matrix(:, dec_val);
    
    % calculating the STD vector for every parameters
    
    Std = zeros(3, 1);
    for kk=1:3
        Std(kk) = (1/2)*(max(Freq2(kk, :)) - min(Freq2(kk, :)));
    end
    
    % Visualizing the reduction of the space parameters - Figure 6
    % NOTE: This section is optional
    
    if ii == 2 || ii==6 || ii == 12
        figure
        scatter3(Freq2(1, :), Freq2(2, :), Freq2(3, :), 'filled', 'blue');
    end
    set(gca, 'fontname', 'times', 'fontsize', 24)
    xlabel('\sigma_{0}', 'fontsize', 25, 'fontname', 'times')
    ylabel('\beta', 'fontsize', 25, 'fontname', 'times')
    zlabel('{\it m}', 'fontsize', 25, 'fontname', 'times')
    axis([150, 300, 0, 2.5, 0, 12]);
    H.YAxis.Color = 'k';
    H.XAxis.Color = 'k';
    H.ZAxis.Color = 'k';
    sprintf("***************************** we passed step %s****************************", num2str(ii))             % for informing user where sequence it is
end

toc;

%% Chi-Squared Goodness of Fit - Table 3

% calculating the chi-squared goodness of fit P-Value (p1, p2, p3) with C.I. of 95%

pd1 = fitdist(Freq2(1, :)','lognormal');
pd2 = fitdist(Freq2(2, :)','weibull');
pd3 = fitdist(Freq2(3, :)','lognormal');

[h1, p1, st1] = chi2gof(Freq2(1, :), 'cdf', pd1, 'alpha', 0.05, 'Emin', 5)
[h2, p2, st2] = chi2gof(Freq2(2, :), 'cdf', pd2, 'alpha', 0.05, 'Emin', 5)
[h3, p3, st3] = chi2gof(Freq2(3, :), 'cdf', pd3, 'alpha', 0.05, 'Emin', 5)

%% Error Ellipses Figure - Figure 4 - Lower Section

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Ellipse of "sigma0" and "Beta"  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
scatter(Freq2(1, :), Freq2(2, :), 'filled', 'r', 'MarkerEdgeColor', "r");
hold on;
plot_ellipse(Freq2(1, :), Freq2(2, :));
H = gca;
xticks([205, 265]);
yticks([-0.5, 1.6]);
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
axis([205, 265, -0.5, 1.6]);
xlabel('\sigma_{0}', 'fontsize', 26, 'fontname', 'times')
ylabel('\beta', 'fontsize', 26, 'fontname', 'times')
set(gca, 'fontname', 'times', 'fontsize', 25)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Ellipse of "sigma0" and "m"  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
scatter(Freq2(1, :), Freq2(3, :), 'filled', 'r', 'MarkerEdgeColor', "r");
hold on;
plot_ellipse(Freq2(1, :), Freq2(3, :));
H = gca;
xticks([205, 265]);
yticks([2.5, 9.5]);
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
axis([205, 265, 2.5, 9.5]);
xlabel('\sigma_{0}', 'fontsize', 26, 'fontname', 'times')
ylabel('{\it m}', 'fontsize', 26, 'fontname', 'times')
set(gca, 'fontname', 'times', 'fontsize', 25)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error Ellipse of "Beta" and "m"  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
scatter(Freq2(2, :), Freq2(3, :), 'filled', 'r', 'MarkerEdgeColor', "r");
hold on;
plot_ellipse(Freq2(2, :), Freq2(3, :));
H = gca;
xticks([-0.5, 1.6]);
yticks([2.5, 9.5]);
H.YAxis.Color = 'k';
H.XAxis.Color = 'k';
axis([-0.5, 1.6, 2.5, 9.5]);
xlabel('\beta', 'fontsize', 26, 'fontname', 'times')
ylabel('{\it m}', 'fontsize', 26, 'fontname', 'times')
set(gca, 'fontname', 'times', 'fontsize', 25)


