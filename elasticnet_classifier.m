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
alpha = .01:.1:1;
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
% tic
% for s = 1:size(x,1)
%     [wt, f] = cwt(x(s,:), Fs);
%     Xcwt(s,:) = wt(1,:)+wt(end,:);
% end
% toc
% X = abs(Xcwt);
%% Visualise stuff...
% TRAIN ELASTIC NET
% tic
% [W,FitInfo] = lassoglm(X(1:m,:),y(1:m),'binomial','NumLambda',5,'CV',5,'Alpha',0.5); % don't add feature of 1's
% toc
% lassoPlot(W,FitInfo,'PlotType','CV');
% legend('show','Location','best') % show legend
% lassoPlot(W,FitInfo,'PlotType','Lambda','XScale','log');
% i = FitInfo.Index1SE;
% w0 = FitInfo.Intercept(i);
% j = W(:,i) ~= 0;
% w = [w0;W(j,i)];
% fprintf('Relevant features: %d of %d\n',sum(j),size(X,2));
% % COMPUTES PREDICTION FOR THE ENTIRE SET AND PLOT RESIDUALS
% yhat = glmval(w,X(:,j),'logit');    % same as util.sigmoid([ones(size(X,1),1) X(:,j)]*w);
% figure;
% histogram(y - yhat)
% title('Residuals from model')
% fprintf('---------------------------------------------------\n')
%% Train Elastic Net
tic
for i = 1:length(alpha)
    [B,FitInfo] = lassoglm(X(1:m,:),y(1:m),'binomial','Lambda', lambda,'Alpha',alpha(i)); % don't add feature of 1's
%     toc
    b0 = FitInfo.Intercept;
    features = B ~= 0;
    b = [b0;B(features)];
    fprintf('Relevant features: %d of %d\n',sum(features),size(X,2));
    yhat = glmval(b,X(1:m,features),'logit');
    predicted = yhat >= thresh;
    train_err(i) = sum(predicted ~= y(1:m)) / numel(y(1:m));

    yhat = glmval(b,X(m+1:end,features),'logit');
    predicted = yhat >= thresh;
    test_err(i) = sum(predicted ~= y(m+1:end)) / numel(y(m+1:end));
end
toc
fprintf('Elastic Net (L1 & L2) - Training Error: %.4f\n',train_err);
% fprintf('Elastic Net (L1 & L2) - Confusion Matrix: \n');
% C = confusionmat(double((predicted)), y(1:m))
fprintf('Elastic Net (L1 & L2) - Test Error: %.4f\n',test_err);
% fprintf('Elastic Net (L1 & L2) - Confusion Matrix: \n');
% C = confusionmat(double((predicted)), y(m+1:end))
fprintf('---------------------------------------------------\n')
plot(alpha,train_err, alpha, test_err)
legend('Training Error', 'Test Error')