%% ==============================================================
%%  ECG Arrhythmia Classification using MIT-BIH and PTBDB
%  --------------------------------------------------------------
%  Description:
%  This script performs ECG signal classification using pre-segmented
%  ECG beats from the MIT-BIH Arrhythmia Dataset and the PTB Diagnostic
%  ECG Database (PTBDB). A fully connected neural network is trained
%  to evaluate classification performance.
%
%  Problem Formulation:
%  - Supervised learning on ECG beat-level data provided in CSV format.
%  - MIT-BIH provides labeled arrhythmia classes.
%  - PTBDB is treated as a binary normal vs abnormal classification task.
%  - Training data combines both datasets to improve generalization,
%    while evaluation is performed exclusively on the MIT-BIH test set.
%% ==============================================================

clc; clear; close all;

%% Dataset Configuration
% Dataset directory defined relative to the project root to ensure
% portability and reproducibility across different environments.

dataFolder = fullfile(pwd, 'data');

assert(isfolder(dataFolder), ...
    'Dataset folder not found. Please ensure the data directory is correctly set.');

%% Data Loading
fprintf('Loading ECG datasets...\n');

mitbih_train = readmatrix(fullfile(dataFolder,'mitbih_train.csv'));
mitbih_test  = readmatrix(fullfile(dataFolder,'mitbih_test.csv'));
ptb_normal   = readmatrix(fullfile(dataFolder,'ptbdb_normal.csv'));
ptb_abnormal = readmatrix(fullfile(dataFolder,'ptbdb_abnormal.csv'));

fprintf('MIT-BIH train: %d x %d\n', size(mitbih_train,1), size(mitbih_train,2));
fprintf('MIT-BIH test : %d x %d\n', size(mitbih_test,1),  size(mitbih_test,2));
fprintf('PTB normal   : %d x %d\n', size(ptb_normal,1),   size(ptb_normal,2));
fprintf('PTB abnormal : %d x %d\n', size(ptb_abnormal,1), size(ptb_abnormal,2));

%% MIT-BIH Data Preparation
X_mit_train = mitbih_train(:,1:end-1);
Y_mit_train = mitbih_train(:,end);

X_mit_test  = mitbih_test(:,1:end-1);
Y_mit_test  = mitbih_test(:,end);

%% PTBDB Data Preparation (Binary Labels)
% 0 = normal, 1 = abnormal

X_ptb = [ptb_normal(:,1:end-1); ptb_abnormal(:,1:end-1)];
Y_ptb = [zeros(size(ptb_normal,1),1); ones(size(ptb_abnormal,1),1)];

%% Dataset Integration
% Training data combines MIT-BIH training set and PTBDB samples.
% Testing is restricted to the MIT-BIH test set for consistency.

XTrain = [X_mit_train; X_ptb];
YTrain = [Y_mit_train; Y_ptb];

XTest = X_mit_test;
YTest = Y_mit_test;

fprintf('Final training set size: %d samples\n', size(XTrain,1));
fprintf('Final testing set size : %d samples\n', size(XTest,1));

%% Signal Normalization
% Each ECG beat is normalized independently to reduce amplitude variance.
% Advanced ECG-specific preprocessing (e.g., baseline wander removal)
% was not applied and is considered a limitation of this approach.

XTrain = normalize(XTrain, 2);
XTest  = normalize(XTest,  2);

%% Label Encoding
YTrain = categorical(YTrain);
YTest  = categorical(YTest);

%% Neural Network Architecture
inputSize = size(XTrain,2);

layers = [
    featureInputLayer(inputSize, "Name", "input")

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(64)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(numel(categories(YTrain)))
    softmaxLayer
    classificationLayer
];

%% Training Configuration and Validation Split
% 10% of the training data is held out for validation.

cv = cvpartition(size(XTrain,1),'HoldOut',0.1);
XVal = XTrain(cv.test,:);
YVal = YTrain(cv.test,:);
XTrain = XTrain(cv.training,:);
YTrain = YTrain(cv.training,:);

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 100);

%% Model Training
fprintf('Starting model training...\n');

net = trainNetwork(XTrain, YTrain, layers, options);

YPredTrain = classify(net, XTrain);
accTrain = mean(YPredTrain == YTrain) * 100;
fprintf('Training Accuracy: %.2f %%\n', accTrain);

%% Model Evaluation
YPred = classify(net, XTest);
accTest = mean(YPred == YTest) * 100;
fprintf('Test Accuracy: %.2f %%\n', accTest);

%% Confusion Matrix and Performance Metrics
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

% Macro-averaged F1 score
cm = confusionmat(YTest, YPred);
precision = diag(cm) ./ sum(cm,2);
recall    = diag(cm) ./ sum(cm,1)';
f1        = 2 * (precision .* recall) ./ (precision + recall);
macroF1   = mean(f1, 'omitnan');

fprintf('Macro F1-score: %.3f\n', macroF1);
