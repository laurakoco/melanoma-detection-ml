
% Author: Laura Kocubinski
% Plot training log 

dir = '/path/to/training/log'

cd(dir)

filename = 'my_model_training.log'

m = csvread(filename,1,0)

epoch = m(:, 1);
acc = m(:, 2);
acc_loss = m(:, 3);
val = m(:, 4);
val_loss = m(:, 5);

figure(1)
subplot(2,2,1)
plot(epoch, acc, epoch, acc_loss)
grid on
xlabel('epoch')
legend('acc','acc\_loss')

subplot(2,2,2)
plot(epoch, val, epoch, val_loss)
grid on
xlabel('epoch')
legend('val','val\_loss')

subplot(2,2,3)
title('Model Accuracy')
plot(epoch, acc, epoch, val)
grid on
xlabel('epoch')
legend('acc','val')

subplot(2,2,4)
title('Model Loss')
plot(epoch, acc_loss, epoch, val_loss)
grid on
xlabel('epoch')
legend('acc\_loss','val\_loss')
