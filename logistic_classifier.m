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
thresh = 0:.1:1;
lambda = 1e-2;
alpha = .5;
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

%% Train Logistic Regression w/ L2 Regularization
% tic
% for i=1:length(thresh)
%     w = 1e-5 * rand(size(X,2),1);
%     options = optimoptions('fminunc','Display','none','SpecifyObjectiveGradient',true,'MaxIterations',1000);
%     [w,~] = fminunc(@(w)(cost(w,X(1:m,:),y(1:m),0)),w,options);
%     posterior = sigmoid(X(1:m,:)*w);
%     predicted = posterior >= thresh(i);
%     train_err(i) = sum(predicted ~= y(1:m)) / numel(y(1:m));
% 
%     posterior = sigmoid(X(m+1:end,:)*w);
%     predicted = posterior >= thresh(i);
%     test_err(i) = sum(predicted ~= y(m+1:end)) / numel(y(m+1:end));
% end
% toc
% fprintf('Logistic Regression (L2) - Training Error: %.4f\n',train_err);
% % fprintf('Logistic Regression (L2) - Confusion Matrix: \n');
% % C = confusionmat(double((predicted)), y(1:m))
% fprintf('Logistic Regression (L2) - Test Error: %.4f\n',test_err);
% % fprintf('Logistic Regression (L2) - Confusion Matrix: \n');
% % C = confusionmat(double((predicted)), y(m+1:end))
% fprintf('---------------------------------------------------\n')
% plot(train_err); hold on; plot(test_err)

thresh = 0.5;
tic
for i=1:length(alpha)
    w = 1e-5 * rand(size(X,2),1);
    options = optimoptions('fminunc','Display','none','SpecifyObjectiveGradient',true,'MaxIterations',1000);
    [w,~] = fminunc(@(w)(cost(w,X(1:m,:),y(1:m),.9)),w,options);
    posterior = sigmoid(X(1:m,:)*w);
    predicted = posterior >= thresh;
    train_err(i) = sum(predicted ~= y(1:m)) / numel(y(1:m));

    posterior = sigmoid(X(m+1:end,:)*w);
    predicted = posterior >= thresh;
    test_err(i) = sum(predicted ~= y(m+1:end)) / numel(y(m+1:end));
end
toc
fprintf('Logistic Regression (L2) - Training Error: %.4f\n',train_err);
% fprintf('Logistic Regression (L2) - Confusion Matrix: \n');
% C = confusionmat(double((predicted)), y(1:m))
fprintf('Logistic Regression (L2) - Test Error: %.4f\n',test_err);
% fprintf('Logistic Regression (L2) - Confusion Matrix: \n');
C = confusionmat(double((predicted)), y(m+1:end))
fprintf('---------------------------------------------------\n')
figure; plot(alpha, train_err); hold on; plot(alpha,test_err)

%% FUNCTION TO COMPUTE SIGMOID
function sig = sigmoid(x)
    sig = 1./(1+exp(x));
end

%% FUNCTION TO COMPUTE COST AND GRADIENT
function [J, grad] = cost(w,X,y,lambda)
    h = sigmoid(X*w);
    J = -sum(y.*log(h)+(1-y).*log(1 - h)) + lambda*sum(w(2:end).^2);
    grad = (X'*(h - y)) + [0; lambda*w(2:end)];
end