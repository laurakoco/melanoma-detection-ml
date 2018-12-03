

# Author: Laura Kocubinski
# Calculate accuracy of AlexNet CNN on testing images

load my_net

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
    
