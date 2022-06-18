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
thresh = 0.5;
lambda = 1e-2;
alpha = 0;
%% Preprocess data
Fs = 173.61;
N = 178;
F = [-N/2:N/2-1]/N;

X = abs(fft(x,[],2)); % compute frequency spectrum
y(y~=1)= 0;

% filter it lol
% tic
% for s = 1:size(X,1)
%     Xf(s,:) = lowpass(X(s,:),15,Fs);
% end
% toc

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

%% Naive Bayes
tic
mdl = fitcnb(X(1:m,:),y(1:m));
toc
[~,posterior,~] = predict(mdl,X(1:m,:));
predicted = posterior(:,1) < posterior(:,2);
train_err = sum(predicted ~= y(1:m)) / numel(y(1:m));
fprintf('Naive Bayes - Training Error: %.4f\n',train_err);
fprintf('Naive Bayes - Confusion Matrix: \n');
C = confusionmat(double((predicted)), y(1:m))
[~,posterior,risk] = predict(mdl,X(m+1:end,:));
predicted = posterior(:,1) < posterior(:,2);
test_err = sum(predicted ~= y(m+1:end)) / numel(y(m+1:end));
fprintf('Naive Bayes - Test Error: %.4f\n',test_err);
fprintf('Naive Bayes - Confusion Matrix: \n');
C = confusionmat(double((predicted)), y(m+1:end))
fprintf('---------------------------------------------------\n')
perfMetric(C)