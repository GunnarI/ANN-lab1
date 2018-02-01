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
patterns = [data1; data2];
patterns = [patterns'; ones(1,200)]; %All data including the bias line

% Create an output matrix
targets = [ones(1,100), -ones(1,100)];   %first data group is 1 and second is -1

% create a weight matrix
[numDims, numInst] = size(patterns);
numClasses = size(targets,1);

%shuffle data by random
shuffle = randperm(200);
patternsShuf = patterns(:,shuffle);
targets = targets(:,shuffle);

%%          3.1.2 Single-layer perceptron

% perceptron learning
% using a batch learning algorithm
% Learning parameters with rate and maximum number of iterations
alpha = 0.7;
eta = 0.001;


etavec = [0.0001,0.001,0.005,0.01,0.1];
epoch = [10 25 50];

weights = randn(1, numDims);

plotId = 0;
for epoch_i = 1:length(epoch)
    
    figure
    if plotId
        plot(patterns(1,1:100),patterns(2,1:100),'bo')
        grid on
        title(['Single Layer - Delta Rule with Epoch = ' num2str(epoch(epoch_i))])
        xlabel('X')
        ylabel('Y')

        hold on
        plot(patterns(1,101:200),patterns(2,101:200),'r+')
    else
        title(['Single Layer (Batch learning) - Epoch = ' num2str(epoch(epoch_i))])
        xlabel('Epoch')
        ylabel('Misclassifications')
    end
        
    [misclass_delta, timevec_delta] = ...
        singleDeltaRule(patternsShuf, targets, eta, weights, epoch(epoch_i), plotId);
    
    if plotId
        figure
        plot(patterns(1,1:100),patterns(2,1:100),'bo')
        grid on
        title(['Single Layer - Perceptron Learning with Epoch = ' num2str(epoch(epoch_i))])
        xlabel('X')
        ylabel('Y')

        hold on
        plot(patterns(1,101:200),patterns(2,101:200),'r+')
    end
    
    [misclass_percept, timevec_percept] = ...
        singlePerceptronLearning(patternsShuf, targets, eta, weights, epoch(epoch_i), plotId);
    
    if ~plotId
        legend('Delta Rule','Perceptron Learning')
    end
    
    
    figure
    if plotId
        plot(patterns(1,1:100),patterns(2,1:100),'bo')
        grid on
        title(['Single Layer - Delta Rule with Epoch = ' num2str(epoch(epoch_i))])
        xlabel('X')
        ylabel('Y')

        hold on
        plot(patterns(1,101:200),patterns(2,101:200),'r+')
    else
        title(['Single Layer (Sequental learning) - Epoch = ' num2str(epoch(epoch_i))])
        xlabel('Epoch')
        ylabel('Misclassifications')
    end
    
    [misclass_delta_seq, timevec_delta_seq] = singleDeltaRuleSeq(patternsShuf, targets, eta, weights, epoch(epoch_i), plotId);
    
    
    [misclass_percept_seq, timevec_percept_seq] = ...
        singlePerceptronLearningSeq(patternsShuf, targets, eta, weights, epoch(epoch_i), plotId);
    
    
    figure
    plot(timevec_delta,'b-')
    hold on
    plot(timevec_percept,'r-')
    hold on
    plot(timevec_delta_seq,'y--')
    hold on
    plot(timevec_percept_seq, 'g--')
    title(['Single Layer (Batch learning) - Epoch = ' num2str(epoch(epoch_i))])
    xlabel('Epoch')
    ylabel('Time')
    legend('Delta Rule','Perceptron Learning', 'Sequental Delta Rule', 'Sequential Perceptron Learning')
    
    
    fprintf('\t\t\tMisclassifications\n')
    fprintf('\t\t\tDelta Rule\tPerceptron Learning\n')
    fprintf('Epoch %d:\t%d\t\t\t%d\n',...
        epoch(epoch_i), misclass_delta, misclass_percept)
end
