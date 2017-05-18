% Project 3  Clustering
%
% Name: Wendong Yuan
% Student Num: 8778806
% Date: Dec 2016
%
% File Name: modelMain.m
%
clear; close all; clc

directory = './playerData/';
format = '.txt';
sumDataset = zeros(300, 10);
probMatrix = zeros(300, 9);

dataset = ['playOnfield1';'playOnfield2';'playOnfield3';...
           'playOnfield4';'playOnfield5';'playOnfield6';...
           'playOnfield7';'playOnfield8'];

% Load datasets
for dataId = 1:8
    fprintf('Loading data file %d ...\n', dataId);
    playerData = load(strcat(directory,dataset(dataId,:),format));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (1,0) right, state = 1
    % (-1,0) left, state = 2
    % (0,-1) top, state = 3
    % (0,1) bottom, state = 4
    % (1,-1) top-right, state = 5
    % (-1,-1) top-left, state = 6
    % (1,1) bottom-right, state = 7
    % (-1,1) top-left, state = 8
    % back, state = 9
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    markovChain = zeros(300, 1600);
    stat = zeros(300, 10);

    pCounter = 1;
    move = 1;
    index = 1;
    subIndex = 1;
    pd_row = size(playerData, 1);
    pd_column = size(playerData, 2);

    for index = 1:pd_row

        if index == pd_row
            break;
        end;

        if playerData(index,1)==0 && playerData(index,2)==0
            subIndex = index;
        end

        if playerData(index+1,1)-playerData(index,1)<-10 || index == pd_row -1
            stat(pCounter, 10) = move - 1;
            pCounter = pCounter +1;
            move = 1;
        else
            state = -1;
            for check = subIndex:index-1
                if playerData(check,1)==playerData(index,1) && playerData(check,2)==playerData(index,2)
                    state = 9;
                    break;
                end
            end

            dir_x = playerData(index+1,1)-playerData(index,1);
            dir_y = playerData(index+1,2)-playerData(index,2);

            if dir_x>0 && dir_y==0 && state~=9
                state = 1;
            end
            if dir_x<0 && dir_y==0 && state~=9
                state = 2;
            end
            if dir_x==0 && dir_y<0 && state~=9
                state = 3;
            end
            if dir_x==0 && dir_y>0 && state~=9
                state = 4;
            end
            if dir_x>0 && dir_y<0 && state~=9
                state = 5;
            end
            if dir_x<0 && dir_y<0 && state~=9
                state = 6;
            end
            if dir_x>0 && dir_y>0 && state~=9
                state = 7;
            end
            if dir_x<0 && dir_y>0 && state~=9
                state = 8;
            end

            markovChain(pCounter, move) = state;
            stat(pCounter, state) = stat(pCounter, state)+1;
            move = move +1;
        end
    end
    
    for row = 1:size(stat,1)
        for column = 1:size(stat,2)
            sumDataset(row, column) = sumDataset(row, column) + stat(row, column);
        end
    end
end

for i = 1:size(sumDataset,1)
    for j = 1:size(sumDataset,2)-1
        probMatrix(i, j) = sumDataset(i,j)/sumDataset(i,10);
    end
end

% set algorithm parameters
TOL = 0.0005;
ITER = 40;
kappa = 3;

% calculate runtime
tic;
% run k-Means
[C, I, iter] = Kmeans(probMatrix, kappa, ITER, TOL);
toc

disp(['k-means iterations: ' int2str(iter)]);

colors = {'red', 'green', 'blue'};

% show plot of clustering
figure;
for i=1:kappa
   hold on, plot3(probMatrix(find(I == i), 1), probMatrix(find(I == i), 4),probMatrix(find(I == i), 7), 'o', 'color', colors{i});
end

% Classify group results
resultSet = zeros(kappa,300);
clusterData = zeros(kappa,1);
for id=1:300
    clusterId = I(id,1);
    clusterData(clusterId,1) = clusterData(clusterId,1)+1;
    resultSet(clusterId,clusterData(clusterId,1))=id;
end;

% Output clustering results
f = fopen('results.txt','wt');
for j=1:kappa
    counter=1;
    fprintf(f,'Group %d (total %d):\n',j,clusterData(j,1));
    while(resultSet(j,counter)~=0)
        fprintf(f,'%d ',resultSet(j,counter));
        counter = counter +1;
    end;
    fprintf(f,'\n');
end;
fclose(f);