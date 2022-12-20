% blacklegRandomForest.m
% Author: Peter Skelsey
% Organisation: The James Hutton Institute
% Date modified: 30/09/2022

% This script trains, tunes, and tests a Random Forest for predicting 
% low / high incidence of blackleg, and calculates predictor importance.

%% Learner

% Load data
T = readtable('yourData.csv');
predictorNames = T.Properties.VariableNames;
predictorNames(end) = []; % Remove response variable name

% Data partitioning
rng(0,'twister');
cvp = cvpartition(T.Incidence,'KFold',10,'Stratify',true);

% Model selection
hypopts = struct('CVPartition',cvp,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'useparallel',false);
t = templateTree('PredictorSelection','interaction-curvature',...
    'SplitCriterion','gdi','NumVariablesToSample','all',...
    'Reproducible',true);
mdl = fitcensemble(T,...
                   'Incidence',...
                   'Learners',t,...
                   'Method','Bag',...
                   'OptimizeHyperparameters',...
                   {'NumLearningCycles','MinLeafSize'},...
                   'HyperparameterOptimizationOptions',hypopts);

% OOB performance
[labelsOOB,scoresOOB] = oobPredict(mdl);
metricsOOB = BinaryConfusionMetrics(T.Incidence,labelsOOB,scoresOOB);

% Predictor importance
impOOB = oobPermutedPredictorImportance(mdl);
[valOOB,idxOOB] = sort(impOOB,'descend');
sortNamesOOB = predictorNames(idxOOB);

%% Local functions

% Performance in predicting labels
function metrics = BinaryConfusionMetrics(yTest,label,score)
    c = confusionmat(yTest,label);
    tn = c(1);
    fn = c(2);
    fp = c(3);
    tp = c(4);
    TPR = tp/(tp+fn);   
    FPR = fp/(fp+tn);   
    TNR = tn/(tn+fp);
    FNR = fn/(fn+tp);
    PPV = tp/(tp+fp);
    NPV =  tn/(tn+fn); 
    acc = (tp+tn)/(tp+tn+fp+fn);
    accBal = (TPR+TNR)/2;
    F1 = 2*(PPV*TPR)/(PPV+TPR);
    [~,~,~,AUROC] = perfcurve(yTest,score(:,2),1);
    metrics = [TPR FPR TNR FNR PPV NPV acc accBal F1 AUROC];
end
