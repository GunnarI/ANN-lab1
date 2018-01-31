clear all
close all
clc

% ANN Project 1
% Create data and imply single layer perceptron learning

%%          3.1.1 create Datasets
% Create 2 datasets of multivariant distribution (with mu and sigma)
% with linearly seperable data (100 points per class)

%rng default;                    %Will always produce the same randon data

% first group of data
mu1 = [1,1];
sigma1 = [0.5,0;0,0.5];
data1 = mvnrnd(mu1,sigma1,100);  %Produces multivariant normal distributed data

%second group of data
mu2 = [-1,-1];
sigma2 = [0.5,0;0,0.5];
data2 = mvnrnd(mu2,sigma2,100);  %Produces multivariant normal distributed data

% combine data into one matrix and add bias line in input
all_data_original = [data1; data2];
all_data_original = all_data_original';
all_data_original = [all_data_original; ones(1,200)]; %All data including the bias line

% Create an output matrix
target = [ones(1,100), -ones(1,100)];   %first data group is 1 and second is -1

% create a weight matrix
[numDims, numInst] = size(all_data_original);
numClasses = size(target,1);
%weights = rand(numClasses, numDims);
%delta_weight = zeros(numClasses, numDims);

%shuffle data by random
shuffle = randperm(200);
all_data = all_data_original(:,shuffle);
target = target(:,shuffle);

%%          Plotting of data points

% figure(1)
% plot(all_data_original(1,1:100),all_data_original(2,1:100),'bo')
% grid on
% title('Data points for classification')
% xlabel('X')
% ylabel('Y')
% 
% hold on
% plot(all_data_original(1,101:200),all_data_original(2,101:200),'r+')

%%          3.1.2 Single-layer perceptron

% perceptron learning
% using a batch learning algorithm
% Learning parameters with rate and maximum number of iterations
alpha = 0.7;
eta = 0.001;
max_iter = 100;     %Max number of iterations
iter = 0;           %iteration counter

etavec = [0.0001,0.001,0.005,0.01,0.1];

for eta_i = 1:length(etavec)
    weights = rand(numClasses, numDims);
    iter = 0; 
    
    figure
    plot(all_data_original(1,1:100),all_data_original(2,1:100),'bo')
    grid on
    title('Data points for classification')
    xlabel('X')
    ylabel('Y')

    hold on
    plot(all_data_original(1,101:200),all_data_original(2,101:200),'r+')

    h = animatedline;
    while iter <= max_iter
        iter = iter + 1;
        % forward pass
        out = weights * all_data;         %Bias part is included in both the weights and the data
        %out = 2./(1 + exp(-y)) - 1;     %transfer to -1 and 1

        %backward pass
        %delta_out = (out - target).*(1 + out).*(1 - out)*0.5;
        delta_out = (target - out);

        %weight update
        weights(1,1) = weights(1,1) + etavec(eta_i).*delta_out*all_data(1,:)';
        weights(1,2) = weights(1,2) + etavec(eta_i).*delta_out*all_data(2,:)';
        weights(1,3) = weights(1,3) + etavec(eta_i).*delta_out*all_data(3,:)';

        %plotting data with bias
        data_weights = weights(1,1:2);
        threshold = -weights(1,3)/(data_weights*data_weights');      %normalised bias is threshold
        norm_weights = sqrt(data_weights*data_weights');

        %Plotting of weights
        hold on
        %axis([-2 2 -2 2])
        clearpoints(h);
        x = 2.*[weights(1),weights(1)];
        y = 2.*[weights(2),weights(2)];
        x2 = 2.*[-weights(2),weights(2)];
        y2 = 2.*[weights(1),-weights(1)];
        xpoints = x*threshold + x2/norm_weights;
        ypoints = y*threshold + y2/norm_weights;
        addpoints(h,xpoints(1),ypoints(1));
        addpoints(h,xpoints(2),ypoints(2));
        drawnow;

        %weights
        pause(0.1)

    end
    hold off


    % Calculate mean square error
    meansquare_error = mean(delta_out.^2)
    % Show misclassifications
    missclass = 0;
    for i = 1:length(target)
        if target(i) > 0 && out(i) < 0
            missclass = missclass + 1;
            missclass_data = i;
        elseif target(i) < 0 && out(i) > 0
            missclass = missclass + 1;
            missclass_data = i;
        end
    end

    missclass
end
