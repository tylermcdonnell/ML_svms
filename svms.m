% Classifying handwritten digits using Support Vector Machines.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load test data.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load digits.mat;

% Reshape data. 28 x 28 images -> 784 vector.
features    = 784;
trainImages = reshape(double(trainImages), features, 60000);
testImages  = reshape(double(testImages), features, 10000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dimensionality Reduction via SVD                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find principal components.
[~, PC] = pcasvd(trainImages);

% Use static number of eigenvectors.
PC = PC(:,1:50);

% Project test and training data.
[~, trainImages] = meannormalize(trainImages);
trainImages = PC' * trainImages;
[~, testImages]  = meannormalize(testImages);
testImages  = PC' * testImages;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Further preprocessing.                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reshape for svm input.
trainImages = trainImages';
testImages  = testImages';

trainLabels = trainLabels';
testLabels  = testLabels';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build learning curve over training set.                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare to baseline PCA + KNN performance.

% Different training sizes for learning curve.
trainSize = [ 10, 100, 500, 1000, 10000, 60000 ];

for N = trainSize
   % Build train set.
   thisTrainSet    = double(trainImages(1:N,:));
   thisTrainLabels = double(trainLabels(1:N));
 
   N
   
   disp('Train')
   testsvm(thisTrainSet, thisTrainLabels,...
           thisTrainSet, thisTrainLabels)
   
   disp('Test')
   
   % Benchmark SVM.
   testsvm(thisTrainSet, thisTrainLabels,...
           testImages, testLabels)
   
   % Benchmark PCA + KNN.
   

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison of Kernel Functions.                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


