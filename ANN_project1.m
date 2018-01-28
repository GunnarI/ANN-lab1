clear all
close all
clc

% ANN Project 1
% Create data which is then used for binary classification

%%          3.1.1 Datasets
% Create 2 datasets of multivariant distribution(with mu and sigma)
% with linearly seperable data (100 points per class)

rng default;
% first group od data
mu1 = [10,4];
sigma1 = [2,1;1,2];
data1 = mvnrnd(mu1,sigma1,100);

%second group of data
mu2 = [-10,-4];
sigma2 = [3,1;1,3];
data2 = mvnrnd(mu2,sigma2,100);

%%          3.1.2 Single-layer perceptron


%%          Plotting

figure(1)
plot(data1(:,1),data1(:,2),'b+')
grid on
xlabel('X')
ylabel('Y')

hold on
plot(data2(:,1),data2(:,2),'r+')

