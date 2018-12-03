
# Author: Laura Kocubinski
# Calculate accuracy of CNN model on testing images

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import csv

# dimensions of our images
img_width, img_height = 128, 128

# load the model we saved
model_file = 'keras_model.h5'
log_file = 'keras_model_training.log'

print(model_file)

model = load_model('models/' + model_file)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# melanoma = 0
# non-melanoma = 1
class_names = ['Melanoma','Non-Melanoma']

dir_mel = '/path/to/test/melanoma'
dir_non_mel = '/path/to/test/non-melanoma'

image_mel = []
image_non_mel = []
test_images = []

for filename in os.listdir(dir_mel):
        # ignore hidden file .DS_STORE
        if filename.lower().endswith('.jpg'):
            img = image.load_img(os.path.join(dir_mel,filename), target_size=(img_width, img_height))
            image_mel.append(img)
            test_images.append(img)
            
for filename in os.listdir(dir_non_mel):
        if filename.lower().endswith('.jpg'):
            img = image.load_img(os.path.join(dir_non_mel,filename), target_size=(img_width, img_height))
            image_non_mel.append(img)
            test_images.append(img)

num_correct = 0

tp = 0
tn = 0
fp = 0
fn = 0

mel_pred = []

# melanoma prediction
for i in range(0,len(image_mel)):
    img = image_mel[i]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    my_image = np.vstack([x])
    pred = model.predict_classes(my_image, batch_size=10)
    mel_pred.append(pred[0][0])
    if mel_pred[i] == 0:
        num_correct = num_correct + 1
        tp = tp + 1
    else:
        fn = fn + 1
 
non_mel_pred = []
    
for i in range(0,len(image_non_mel)):
    img = image_non_mel[i]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    my_image = np.vstack([x])
    pred = model.predict_classes(my_image, batch_size=10)
    non_mel_pred.append(pred[0][0])
    if non_mel_pred[i] == 1:
        num_correct = num_correct + 1
        tn = tn + 1
    else:
        fp = fp + 1
    
acc = np.divide( ( float(tp) + float(tn) ), ( float(tp) + float(tn) + float(fp) + float(fn)) )
sens = np.divide( float(tp), ( float(tp) + float(fn) ) )
spec = np.divide( float(tn), ( float(tn) + float(fp) ) )

print "Accuracy: " + str(acc)
print "Sensitivity: " + str(sens)
print "Specificity: " + str(spec)

