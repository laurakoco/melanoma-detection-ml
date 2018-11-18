
# Author: Laura Kocubinski
# Build CNN in Keras to Detect Melanoma

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Sequential, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

train_data_dir = '/path/to/train'
validation_data_dir = '/path/to/validation'

nb_train_samples = 1505
nb_validation_samples = 214

batch_size = 16
nb_epochs = 10
nb_fc_neurons = 256
nb_filter = 32
nb_conv2d = 3

img_width, img_height = 128, 128
    
log_name = 'my_model_training.log'
model_name = 'my_model.h5'
    
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = Sequential()
    
model.add(Conv2D(nb_filter, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Conv2D(nb_filter*2, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(nb_filter*4, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Flatten())
model.add(Dense(nb_fc_neurons))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')

best_model_val_acc = ModelCheckpoint('best_model_val_acc',monitor='val_acc',
                                    mode = 'max', verbose=1, save_best_only=True)
best_model_val_loss = ModelCheckpoint('best_model_val_loss',monitor='val_loss',
                                    mode = 'min', verbose=1, save_best_only=True)

# early stopping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=2,
                                               verbose=0,
                                               mode='auto',
                                               baseline=None,
                                               restore_best_weights=False
                                               )

# create csv with training data log
csv_logger = CSVLogger('models/'+log_name)

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epochs,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[csv_logger, best_model_val_acc, best_model_val_loss]
    )

# save model
model.save('models/' + model_name)
