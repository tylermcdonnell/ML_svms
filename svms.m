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

% % Different training sizes for learning curve.
% trainSize = [ 10, 25, 50, 100, 500, 1000, 10000, 60000 ];
% results   = zeros(1, length(trainSize));
% 
% for i = 1:length(trainSize)
%    N = trainSize(i);
%     
%    % Build train set.
%    thisTrainSet    = double(trainImages(1:N,:));
%    thisTrainLabels = double(trainLabels(1:N));
%    
%    % Test SVM.
%    results(i) = testsvm(thisTrainSet, thisTrainLabels,...
%                         testImages, testLabels)
% end
% 
% knnTrainSizes = [ 10, 250, 7000, 60000 ];  
% knnResults    = [ 0.0, 0.80, 0.90, 0.94 ];
% 
% % Plot results.
% figure()
% plot(trainSize, results,...
%      knnTrainSizes, knnResults);
% xlabel('Training Examples');
% ylabel('Accuracy');
% curtick = get(gca, 'XTick');
% set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
% axis([0 60000 0 1])
% legend('SVM-Rbf', 'KNN')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build accuracy curve over different dimensionalities.                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Different dimensionalities.
% trainingSize         = [ 500, 5000, 10000, 20000, 60000 ];
% testDimensionalities = [ 2, 4, 8, 16, 32, 64, 128, 256, 512] ;
% results = zeros(length(trainingSize),length(testDimensionalities));
% for i = 1:length(trainingSize)
% for j = 1:length(testDimensionalities)    
%     T = testDimensionalities(j);
%     
%     % Reshape for PCA input.
%     reshapedTrain  = trainImages(1:trainingSize(i),:)';
%     reducedTrainLabels = trainLabels(1:trainingSize(i));
%     reshapedTest  = testImages';
%    
%     % Find principal components.
%     [~, PC] = pcasvd(reshapedTrain);
% 
%     % Use static number of eigenvectors.
%     PC = PC(:,1:T);
% 
%     % Project test and training data.
%     [~, projectedTrainSet] = meannormalize(reshapedTrain);
%     projectedTrainSet = PC' * projectedTrainSet;
%     [~, projectedTestSet]  = meannormalize(reshapedTest);
%     projectedTestSet  = PC' * projectedTestSet;
%     
%     % Reshape for SVM input.
%     projectedTrainSet = projectedTrainSet';
%     projectedTestSet  = projectedTestSet';
%     
%     results(i,j) = testsvm(projectedTrainSet, reducedTrainLabels,...
%                            projectedTestSet,  testLabels)
%    
% end
% end
% 
% knnDimensionalities = [ 0 20 200 512];  
% knnResults = [ 0.0 0.92 0.905 0.90 ];
% 
% % Plot results.
% figure()
% plot(testDimensionalities, results(1,:),...
%      testDimensionalities, results(2,:),...
%      testDimensionalities, results(3,:),...
%      testDimensionalities, results(4,:),...
%      knnDimensionalities, knnResults);
% xlabel('Eigenvectors');
% ylabel('Accuracy');
% %curtick = get(gca, 'XTick');
% %set(gca, 'XTickLabel', cellstr(num2str(curtick(:))));
% axis([0 512 0 1])
% legend('SVM N=500',...
%        'SVM N=5000',...
%        'SVM N=10000',...
%        'SVM N=20000',...
%        'KNN N=5000')
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time comparison.                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
trainSizes = [ 5000 10000 25000 60000 ];
testSizes  = [ 5000 10000 ];

trainTimes = zeros(length(trainSizes), length(testSizes));
testTimes  = zeros(length(trainSizes), length(testSizes));
accuracy   = zeros(length(trainSizes), length(testSizes));

trainTimesKnn = zeros(length(trainSizes), length(testSizes));
testTimesKnn  = zeros(length(trainSizes), length(testSizes));
accuracyKnn   = zeros(length(trainSizes), length(testSizes));
for i = 1:length(trainSizes)
for j = 1:length(testSizes)
    % Build train and test sets.
    Ntrain = trainSizes(i);
    Ntest  = testSizes(j);
    thisTrainSet      = double(trainImages(1:Ntrain,:));
    thisTrainLabels   = double(trainLabels(1:Ntrain));
    thisTestSet       = double(testImages(1:Ntest,:));
    thisTestSetLabels = double(testLabels(1:Ntest));
    
    % SVM
%     tic;
%     t = templateSVM('Standardize', 1,... 
%                 'KernelFunction', 'rbf',... 
%                 'KernelScale','auto');
%     svmClassifier = fitcecoc(thisTrainSet, thisTrainLabels, 'Learners', t);
%     trainTimes(i,j) = toc;
%     
%     tic;
%     results = predict(svmClassifier, thisTestSet);
%     correct = nnz((results == thisTestSetLabels));
%     accuracy(i,j) = double(correct) / double(length(thisTestSetLabels));
%     testTimes(i,j) = toc;
    
    %KNN
    t = 
end
end

    
trainTimes
testTimes
accuracy
