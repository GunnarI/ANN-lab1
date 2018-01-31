close all
clear all
clc

% ANN Project 1
% Create data and imply single layer perceptron learning

%%          3.1.1 create Datasets
% Create 2 datasets of multivariant distribution (with mu and sigma)
% with linearly seperable data (100 points per class)

%rng default;                    %Will always produce the same randon data

% first group of data
mu1 = [5,5];
sigma1 = [2,1;1,2];
data1 = mvnrnd(mu1,sigma1,100);  %Produces multivariant normal distributed data

%second group of data
mu2 = [-5,-5];
sigma2 = [2,1;1,2];
data2 = mvnrnd(mu2,sigma2,100);  %Produces multivariant normal distributed data

% combine data into one matrix and add bias line in input
all_data = [data1; data2];
all_data = all_data';
all_data = [all_data; ones(1,200)]; %All data including the bias line

% Create an output matrix
target = [ones(1,100), -ones(1,100)];   %first data group is 1 and second is -1

% create a weight matrix
[numDims, numInst] = size(all_data);
numClasses = size(target,1);
weights = zeros(numClasses, numDims);
delta_weights = zeros(numClasses, numDims);

%%          Plotting of data points

figure(1)
plot(all_data(1,1:100),all_data(2,1:100),'b+')
grid on
title('Data points for classification')
xlabel('X')
ylabel('Y')

hold on
plot(all_data(1,101:200),all_data(2,101:200),'r+')

%%          3.1.2 Single-layer perceptron

% perceptron learning
% using a batch learning algorithm
% Learning parameters with rate and maximum number of iterations
alpha = 0.7;
eta = 0.001;
max_iter = 100;     %Max number of iterations
iter = 0;           %iteration counter

while iter <= max_iter
    iter = iter + 1;
    
    % forward pass
    out = weights*all_data;                         %Bias part is included in both the weights and the data
    %out = 2./(1 + exp(-out)) - 1;                    %transferfunction to -1 and 1, NOT for delta function
    delta_weights = -eta.*(out - target)*all_data';    %change in weight function
    
    %weight update
    weights = weights + delta_weights;
    
    %plotting data with bias
    data_weights = weights(1,1:2);
    threshold = weights(1,3)/(data_weights*data_weights');      %normalised bias is threshold
    norm_weights = sqrt(data_weights*data_weights');
    
    %Plotting of weights
    hold on
    axis([-10 10 -10 10])
    x = [weights(1),weights(1)];
    y = [weights(2),weights(2)];
    x2 = [weights(1),-weights(1)];
    y2 = [-weights(2),weights(2)];
    line(x*threshold + x2/norm_weights, y*threshold + x2/norm_weights, 'Color','black','LineStyle','-')

    weights
    pause(0.1)

end
hold off


% Calculate mean square error
meansquare_error = mean(delta_weights.^2)

% Show misclassifications
missclass = 0;
for i = 1:length(target)
    if i <= length(target)/2 && out(i) < 0
        missclass = missclass + 1;
        missclass_data = i;
    elseif i > length(target)/2 && out(i) > 0
        missclass = missclass + 1;
        missclass_data = i;
    end
end

missclass
