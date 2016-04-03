% Classifying handwritten digits using Support Vector Machines.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load test data.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load digits.mat;

% Reshape data. 28 x 28 images -> 784 vector.
features    = 784;
trainImages = reshape(trainImages, features, 60000);
testImages  = reshape(testImages, features, 10000);

% Reshape for svm input.
testImages  = testImages';
trainImages = trainImages';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build learning curve over training set.                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare to baseline PCA + KNN performance.

% Different training sizes for learning curve.
trainSize = [ 10, 100, 500, 1000, 10000 ];

for N = trainSize
   % Build train set.
   thisTrainSet    = trainImages(1:N,:);
   thisTrainLabels = trainLabels(1:N);
   
   N
   
   % Benchmark SVM.
   testsvm(thisTrainSet, thisTrainLabels, testImages, testLabels)
   
   % Benchmark PCA + KNN.
   

end

