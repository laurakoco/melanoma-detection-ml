
% Author: Laura Kocubinski
% Train AlexNet CNN to Detect Melanoma

net = alexnet

digitDatasetPath = '/path/to/images/'

imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'FileExtensions',{'.bmp','.jpg'},'LabelSource','foldernames');

[imdsTrain,imdsTest,imdsValidation] = splitEachLabel(imds,0.7,0.2,0.1,'randomized');

imageSize = [227 227 3];

augimdsTrain = augmentedImageSource(imageSize, imdsTrain);
augimdsTest = augmentedImageSource(imageSize, imdsTest);
augimdsValidation = augmentedImageSource(imageSize, imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-5, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

netTransfer = trainNetwork(augimdsTrain,layers,options);

my_net = netTransfer 
save my_net

[YPred_Val,probs] = classify(netTransfer,augimdsValidation);
accuracy_val = mean(YPred_Val == imdsValidation.Labels)

[YPred_Test,probs] = classify(netTransfer,augimdsTest);
acc_test = mean(YPred_Test == imdsTest.Labels)
    
tp = 0;
tn = 0;
fp = 0;
fn = 0;

for i=1:length(imdsTest.Labels)
    if imdsTest.Labels(i) == 'Melanoma'
        if YPred_Test(i) == 'Melanoma'
            tp = tp + 1;
        end
        if YPred_Test(i) == 'Non-Melanoma'
            fn = fn + 1;
        end
    end
    if imdsTest.Labels(i) == 'Non-Melanoma'
        if YPred_Test(i) == 'Melanoma'
            fp = fp + 1;
        end
        if YPred_Test(i) == 'Non-Melanoma'
            tn = tn + 1;
        end
    end
    
end
    
acc = ( ( tp + tn ) / ( tp + tn + fp + fn ) )
sens = ( tp / ( tp + fn ) )
spec = ( tn / ( tn + fp ) )
