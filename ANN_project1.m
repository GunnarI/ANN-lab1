clear all
close all
clc

% ANN Project 1
% Create data which is then used for perceptron learning

%%          3.1.1 create Datasets
% Create 2 datasets of multivariant distribution(with mu and sigma)
% with linearly seperable data (100 points per class)

%rng default;                    %Will always produce the same randon data

% first group of data
mu1 = [10,4];
sigma1 = [10,1;1,10];
data1 = mvnrnd(mu1,sigma1,100);

%second group of data
mu2 = [-10,-4];
sigma2 = [10,1;1,10];
data2 = mvnrnd(mu2,sigma2,100);

% combine data into one matrix with bias line in input
all_data = [data1; data2];
all_data = all_data';
bias = ones(1,200);
all_data = [all_data; bias];

% Create an output matrix
target = [ones(1,100), -ones(1,100)];

% create a weight matrix
[numDims, numInst] = size(all_data);
numClasses = size(target,1);
weights = zeros(numClasses, numDims);
delta_weight = zeros(numClasses, numDims);

%%          Plotting

figure(1)
plot(all_data(1,1:100),all_data(2,1:100),'b+')
grid on
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
max_iter = 100;
%min_error = 0;

% actual output of each epoch
out = zeros(1,200);
iter = 0;

while iter <= max_iter
    iter = iter + 1;
    
    %for i = 1:length(all_data)
        
        % forward pass
        y = all_data(1,:) * weights(1,1)...
            + all_data(2,:) * weights(1,2);...
            %+ all_data(3,:) * weights(1,3);                               %bias part
        out = 2./(1 + exp(-y)) - 1;
        
        %backward pass
        delta_out = (out - target).*(1 + out).*(1 - out)*0.5;
        
        %weight update
        delta_weight(1,1) = (delta_weight(1,1) .* alpha) - (delta_out * all_data(1,:)') .* (1-alpha);
        weights(1,1) = weights(1,1) + delta_weight(1,1) .* eta;
        delta_weight(1,2) = (delta_weight(1,2) .* alpha) - (delta_out * all_data(2,:)') .* (1-alpha);
        weights(1,2) = weights(1,2) + delta_weight(1,2) .* eta;
        %bias weight update
        delta_weight(1,3) = (delta_weight(1,3) .* alpha) - (delta_out * all_data(3,:)') .* (1-alpha);
        weights(1,3) = weights(1,3) + delta_weight(1,3) .* eta;
        
    %end

%Plotting of weights
hold on
axis([-20 20 -20 20])
pointx = [(-20*weights(1,1) + weights(1,3)), (20*weights(1,1) + weights(1,3))];
pointy = [20*weights(1,2),-20*weights(1,2)];
line(pointx,pointy,'Color','black','LineStyle','-')

pause(0.1)
end
hold off

% Calculate mean square error
meansquare_error = mean(delta_out.^2)
% Show misclassifications
missclass = 0;
for i = 1:length(target)
    if i <= 100 && out(i) < 0
        missclass = missclass + 1;
        missclass_data = i;
    elseif i > 100 && out(i) > 0
        missclass = missclass + 1;
        missclass_data = i;
    end
end

missclass

%% delta learning



