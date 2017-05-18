% Project 2  Classification
%
% Name: Wendong Yuan
% Student Num: 8778806
% Date: Nov 2016
%
% File Name: mainClassification.m
%
clear ; close all; clc
directory = './observed/';
format = '.mat';

% Name all the datasets
smallDataset = ['classify_d3_k2_saved1';'classify_d3_k2_saved2';'classify_d3_k2_saved3';...
                'classify_d4_k3_saved1';'classify_d4_k3_saved2';'classify_d5_k3_saved1';...
                'classify_d5_k3_saved2'];
            
bigDataset = ['classify_d99_k50_saved1';'classify_d99_k50_saved2';...
              'classify_d99_k60_saved1';'classify_d99_k60_saved2'];       

fprintf('Start Training...\n\n');

% Load datasets
for dataId = 1:11
    datasetName = '';
    if dataId<=7
        load(strcat(directory,smallDataset(dataId,:),format));
        datasetName = smallDataset(dataId,:);
    else
        load(strcat(directory,bigDataset(dataId-7,:),format));
        datasetName = bigDataset(dataId-7,:);
    end
    
    fprintf('Training Dataset [%s]...\n',datasetName);
    % Count time
    t1=clock;
    C = 1;
    portion = 0.8; % Ratio 4:1

    % Size of the dataset matrix
    row_c = size(class_1, 1);
    column_c = size(class_1, 2);

    % Training matrix
    M1=class_1(:,1:column_c*portion);
    M2=class_2(:,1:column_c*portion);
    M = [M1,M2];

    % Results matrix for training
    N1 = zeros(1,column_c*portion);
    N2 = ones(1,column_c*portion);
    N = [N1,N2];

    % Train model
    model = modelTraining(M', N', C, @gaussianKernel);

    test_portion = 1-portion;
    
    T1=class_1(:,(column_c*portion+1):end);
    T2=class_2(:,(column_c*portion+1):end);
    T = [T1,T2];

    R1 = zeros(1,int16(column_c*test_portion));
    R2 = ones(1,int16(column_c*test_portion));
    R = [R1,R2];
    R = R';
    
    % Test model
    p = modelTesting(model, T');
    comp = (p == R);

    sample = size(R,1);
    preError = sample - sum(comp,1);

    t2=clock;
    
    % Print out performance stat
    fprintf('Predict Errors: %d out of %d samples\n', preError, sample);
    fprintf('Test Accuracy: %f\n', double(sum(comp,1)/sample) * 100);
    fprintf('Runtime: %d\n', etime(t2,t1));
    save(strcat('./models/MODEL-',datasetName), '-struct', 'model');
    fprintf('Model for %s has been generated!\n\n', datasetName);
end

fprintf('End Training\n');


