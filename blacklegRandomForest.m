% blacklegRandomForest.m
% Author: Peter Skelsey
% Organisation: The James Hutton Institute, Dundee, UK
% Date modified: 28/12/2022

% This script develops and interprets a Random Forest model for predicting
% incidence of blackleg at the landscape-scale

%% Learner

% Load data
T = readtable('yourData.csv');
predictorNames = T.Properties.VariableNames;
predictorNames(end) = [];

% Random Forest
rng(0,'twister');
hypopts = struct('AcquisitionFunctionName','expected-improvement-plus',...
    'useparallel',false,'ShowPlots',false);
t = templateTree('PredictorSelection','interaction-curvature',...
    'MaxNumSplits',20,'SplitCriterion','gdi','Reproducible',true);
mdl = fitcensemble(T,'Incidence','Learners',t,'Method','Bag',...
   'OptimizeHyperparameters',{'NumLearningCycles','MinLeafSize',...
   'NumVariablesToSample'},'HyperparameterOptimizationOptions',hypopts);

% Predictive performance
[labelsResub,scoresResub] = resubPredict(mdl);
metricsResub = BinaryConfusionMetrics(T.Incidence,labelsResub,scoresResub);
[labelsOOB,scoresOOB] = oobPredict(mdl);
metricsOOB = BinaryConfusionMetrics(T.Incidence,labelsOOB,scoresOOB);

% Predictor importance
impOOB = oobPermutedPredictorImportance(mdl);

%% Local functions

% Performance in predicting labels
function metrics = BinaryConfusionMetrics(yTest,labels,scores)
    c = confusionmat(yTest,labels);
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
    [~,~,~,AUROC] = perfcurve(yTest,scores(:,2),1);
    metrics = [TPR FPR TNR FNR PPV NPV acc accBal F1 AUROC];
end
