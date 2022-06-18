close all; clear all; clc;
load data.mat

rng(1) % seed the RNG for consistency

x = [X(y==1,:); X(y==5,:)];
y = [y(y==1); y(y==5)];

m = size(x,1);
shuffle = randperm(m);
x = x(shuffle,:);
y = y(shuffle);

pct = 0.8; 
m = round(pct*m);
%% Preprocess data
Fs = 173.61;
N = 178;
F = [-N/2:N/2-1]/N;

X = abs(fft(x,[],2)); % compute frequency spectrum
y(y~=1)= 0;

% Take the IMF
% tic
% for s = 1:size(x,1)
%     [IMF, res] = emd(x(s,:),'Display',0);
% %     Ximf(s,:) = sum(IMF(:,1:2)',1);
%     Ximf(s,:) = IMF(:,1)' + IMF(:,end)';
% end
% toc
% X = Ximf;

% Take the CWT
tic
for s = 1:size(x,1)
    [wt, f] = cwt(x(s,:), Fs);
    Xcwt(s,:) = wt(1,:)+wt(end,:);
end
toc
X = abs(Xcwt);
%% SVM Classifier
tic
SVMModel = fitcsvm(X(1:m,:),y(1:m),'KernelFunction','linear','KernelScale',6.2014,'BoxConstraint',0.001);
% SVMModel = fitcsvm(X(1:m,:),y(1:m),'OptimizeHyperparameters','auto');
toc
y_predict = SVMModel.predict(X(1:m,:));
train_err = sum(y_predict ~= y(1:m)) / numel(y(1:m));
fprintf('SVM - Training Error: %.4f\n',train_err);
fprintf('SVM - Confusion Matrix: \n');
C = confusionmat(double((y_predict)), y(1:m))
y_val = SVMModel.predict(X(m+1:end,:));
test_err = sum(y_val ~= y(m+1:end)) / numel(y(m+1:end));
fprintf('SVM - Test Error: %.4f\n',test_err);
fprintf('SVM - Confusion Matrix: \n');
C = confusionmat(double((y_val)), y(m+1:end))
fprintf('---------------------------------------------------\n')