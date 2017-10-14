rng(1);
addpath('utils/');
addpath('npy');
addpath('mex');

%% load data
dataset = 'sift1m';
% dataset = 'labelme';
% dataset = 'oxford';
RESUME = 1; 
model_filename = 'record/171010_101167_sift1m_M8_K256.mat';
% model_filename = 'record/171009_222553_oxford_M4_K256.mat'
% model_filename = 'record/171009_232642_labelme_M8_K256.mat';


LOG = logtime();



if strcmp(dataset, 'labelme')
    Xtrain = single(readNPY('../labelme_train.npy'));
    Xbase = Xtrain;
    Xquery = single(readNPY('../labelme_test.npy'));
    nns = readNPY('../labelme_nns.npy') + 1;
elseif strcmp(dataset, 'sift1m')
    data = load('../../data/sift/sift1m_pre.mat');
    Xtrain = data.trdata_pre';
    Xbase = data.base_pre';
    Xquery = data.query_pre';
    nns = ivecs_read('../../data/sift/sift_groundtruth.ivecs')' + 1;
elseif strcmp(dataset, 'oxford')
%     Xtrain = single(readNPY('oxford_train.npy'));
%     Xquery = single(readNPY('oxford_test.npy'));
    Xtrain = single(readNPY('oxford_train.npy'));
    Xquery = single(readNPY('oxford_test.npy'));
    Xbase = Xtrain;
    
    nns = knnsearch(Xbase, Xquery, 'K', 10);
end 

A = load('tmp.mat');


model = A.model;
disp(model);
hash = A.hash;
fprintf('T = 1...\n');
tic;
sccq_evaluate(model, Xquery, hash, nns(:, 1:1), [1, 2, 5, 10, 20, 50, 100, 200, 500]);
t = toc;
       
