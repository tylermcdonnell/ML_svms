function [ accuracy ] = testsvm( trainSet,...
                                 trainLabels,...
                                 testSet,...
                                 testLabels )
%testSVM Trains SVM and classifies set test. Returns accuracy.  
%   Arguments:
%
%   trainSet        - Train set.
%   trainLabels     - Labels for train set.
%   testSet         - Test set.
%   testLabels      - Labels for test set.

% Ensure that data is float.
trainSet = double(trainSet);
testSet  = double(testSet);

% Train SVM.
t = templateSVM('Standardize', 1,... 
                'KernelFunction', 'rbf',... 
                'KernelScale','auto');
svmClassifier = fitcecoc(trainSet, trainLabels, 'Learners', t);

disp('Classifying...')
% Classify test set.
results = predict(svmClassifier, testSet);

% Compute accuracy.
correct  = nnz((results == testLabels));
accuracy = double(correct) / double(length(testLabels));

end

