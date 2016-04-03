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
svmClassifier = fitcecoc(trainSet, trainLabels);

% Classify test set.
results = predict(svmClassifier, testSet);

% Compute accuracy.
correct  = nnz((results == testLabels'));
accuracy = double(correct) / double(length(testLabels));

end
