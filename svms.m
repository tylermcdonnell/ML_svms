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
trainSize = [ 10, 25, 50, 100, 500, 1000, 10000, 60000 ];
results   = zeros(1, length(trainSize));

for i = 1:length(trainSize)
   N = trainSize(i);
    
   % Build train set.
   thisTrainSet    = double(trainImages(1:N,:));
   thisTrainLabels = double(trainLabels(1:N));
   
   % Test SVM.
   results(i) = testsvm(thisTrainSet, thisTrainLabels,...
                        testImages, testLabels)
end

knnTrainSizes = [ 10, 250, 7000, 60000 ];  
knnResults    = [ 0.0, 0.80, 0.90, 0.94 ];

% Plot results.
figure()
plot(trainSize, results,...
     knnTrainSizes, knnResults);
xlabel('Training Examples');
ylabel('Accuracy');
curtick = get(gca, 'XTick');
set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
axis([0 60000 0 1])
legend('SVM-Rbf', 'KNN')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison of Kernel Functions.                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


